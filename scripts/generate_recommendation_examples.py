"""Generate anonymized qualitative examples for GPTRec and PPO policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from gptrec.data.reward_dataset import KIONRewardDataset, RewardEventStore
from gptrec.models.ppo_policy import (
    GPTRecActorCritic,
    PPOPolicyConfig,
    append_policy_step,
)
from gptrec.models.reward_ensemble import RewardModelEnsemble
from gptrec.training.ppo_trainer import (
    ItemFeatureLookup,
    PPOTrainingConfig,
    _candidate_ids,
    _choose_device,
    _feedback_vector,
    _load_base,
    _policy_distribution,
    _raw_reward,
    _reward_checkpoints,
)

RolloutMode = Literal["slate", "feedback"]
COMPONENTS = (
    "completion_utility",
    "duration_utility",
    "return_probability",
    "epistemic_uncertainty",
)


@dataclass(frozen=True, slots=True)
class PolicySpec:
    key: str
    label: str
    mode: RolloutMode
    checkpoint: Path | None


POLICIES = (
    PolicySpec("base", "GPTRec-base", "slate", None),
    PolicySpec(
        "slate_beta_05",
        "PPO-slate β=0.5, update 50",
        "slate",
        Path("artifacts/checkpoints/ppo_high_kl/slate/update_050.pt"),
    ),
    PolicySpec(
        "feedback_beta_05",
        "PPO-feedback β=0.5, update 50",
        "feedback",
        Path("artifacts/checkpoints/ppo_high_kl/feedback/update_050.pt"),
    ),
    PolicySpec(
        "slate_beta_005_long",
        "PPO-slate β=0.05, update 6000",
        "slate",
        Path("artifacts/checkpoints/ppo_final/slate/update_6000.pt"),
    ),
)


def _load_policy(
    spec: PolicySpec,
    *,
    base_checkpoint: Path,
    device: torch.device,
) -> GPTRecActorCritic:
    if spec.checkpoint is None:
        return GPTRecActorCritic.from_gptrec_checkpoint(
            base_checkpoint,
            PPOPolicyConfig(),
            map_location=device,
        ).to(device)
    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"missing policy checkpoint: {spec.checkpoint}")
    checkpoint = torch.load(spec.checkpoint, map_location=device, weights_only=False)
    policy = GPTRecActorCritic.from_gptrec_checkpoint(
        base_checkpoint,
        PPOPolicyConfig(**checkpoint.get("policy_config", {})),
        map_location=device,
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    return policy


@torch.no_grad()
def _generate_batch(
    policy: GPTRecActorCritic,
    reference: torch.nn.Module,
    reward_model: RewardModelEnsemble,
    feature_lookup: ItemFeatureLookup,
    raw_batch: dict[str, Tensor],
    config: PPOTrainingConfig,
    *,
    mode: RolloutMode,
    device: torch.device,
) -> dict[str, Tensor]:
    input_ids = raw_batch["input_ids"].to(device)
    attention_mask = raw_batch["attention_mask"].to(device)
    feedback = torch.zeros(
        *input_ids.shape,
        policy.config.feedback_feature_dim,
        device=device,
    )
    actions_by_step: list[Tensor] = []
    reward_by_step: list[Tensor] = []
    components_by_step = {name: [] for name in COMPONENTS}
    for _ in range(config.horizon):
        actor_logits, _ = policy(input_ids, attention_mask, feedback)
        reference_logits = reference.next_item_logits(input_ids, attention_mask)
        candidates = _candidate_ids(reference_logits, input_ids, config.candidate_count)
        actor_log_probabilities, _ = _policy_distribution(actor_logits, candidates)
        action_indices = actor_log_probabilities.argmax(dim=-1)
        actions = candidates.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        content_types, numeric = feature_lookup.for_actions(actions)
        predictions = reward_model.predict(
            input_ids,
            attention_mask,
            actions,
            content_types,
            numeric,
        )
        actions_by_step.append(actions.cpu())
        reward_by_step.append(_raw_reward(predictions, config.reward_weights).cpu())
        for name in COMPONENTS:
            components_by_step[name].append(predictions[name].cpu())
        input_ids, attention_mask, feedback = append_policy_step(
            input_ids,
            attention_mask,
            feedback,
            actions,
            _feedback_vector(predictions, mode=mode),
        )
    return {
        "recommendations": torch.stack(actions_by_step, dim=1),
        "rewards": torch.stack(reward_by_step, dim=1),
        **{
            name: torch.stack(values, dim=1)
            for name, values in components_by_step.items()
        },
    }


def _item_catalog(items_path: Path, vocabulary_path: Path) -> dict[int, dict[str, Any]]:
    items = pd.read_parquet(
        items_path,
        columns=["item_id", "title", "release_year", "genres"],
    )
    vocabulary = pd.read_parquet(
        vocabulary_path,
        columns=["item_id", "item_token_id"],
    )
    merged = vocabulary.merge(items, on="item_id", how="left", validate="one_to_one")
    result: dict[int, dict[str, Any]] = {}
    for row in merged.itertuples(index=False):
        title = str(row.title) if pd.notna(row.title) else f"item {row.item_id}"
        year = int(row.release_year) if pd.notna(row.release_year) else None
        genres = str(row.genres) if pd.notna(row.genres) else "жанр не указан"
        result[int(row.item_token_id)] = {
            "item_id": int(row.item_id),
            "title": title,
            "year": year,
            "genres": genres,
        }
    return result


def _item_text(token: int, catalog: dict[int, dict[str, Any]]) -> str:
    item = catalog.get(token)
    if item is None:
        return f"item token {token}"
    has_year = item["year"] is not None and str(item["year"]) in item["title"]
    year = f" ({item['year']})" if item["year"] is not None and not has_year else ""
    return f"{item['title']}{year}"


def _rank(target: int, recommendations: list[int]) -> int | None:
    try:
        return recommendations.index(target) + 1
    except ValueError:
        return None


def _overlap(left: list[int], right: list[int]) -> int:
    return len(set(left).intersection(right))


def _choose_index(
    candidates: list[int],
    scores: np.ndarray,
    used: set[int],
) -> int:
    available = [index for index in candidates if index not in used]
    pool = available or candidates
    if not pool:
        return int(scores.argmax())
    return max(pool, key=lambda index: float(scores[index]))


def _select_examples(
    results: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str, int]]:
    base = results["base"]
    slate = results["slate_beta_05"]
    feedback = results["feedback_beta_05"]
    collapsed = results["slate_beta_005_long"]
    targets = np.asarray(base["targets"])
    base_rec = base["recommendations"]
    slate_rec = slate["recommendations"]
    feedback_rec = feedback["recommendations"]
    low_rec = collapsed["recommendations"]
    base_reward = np.asarray(base["mean_rewards"])
    slate_reward = np.asarray(slate["mean_rewards"])
    reward_gain = slate_reward - base_reward
    base_hit = np.asarray([target in rec for target, rec in zip(targets, base_rec, strict=True)])
    slate_hit = np.asarray(
        [target in rec for target, rec in zip(targets, slate_rec, strict=True)]
    )
    used: set[int] = set()
    selected: list[tuple[str, str, str, int]] = []

    preserved_candidates = np.flatnonzero(base_hit & slate_hit & (reward_gain > 0)).tolist()
    preserved = _choose_index(preserved_candidates, reward_gain, used)
    used.add(preserved)
    selected.append(
        (
            "Reward вырос, target сохранился",
            "Пример улучшения оценки RM без потери попадания реального следующего item.",
            "Обе ранние PPO-policy сохраняют наблюдавшийся target и при этом формируют "
            "содержательно связанный список. Это наиболее простой случай, где offline-метрика "
            "и качественная интерпретация не противоречат друг другу.",
            preserved,
        )
    )

    tradeoff_candidates = np.flatnonzero(base_hit & ~slate_hit & (reward_gain > 0)).tolist()
    tradeoff = _choose_index(tradeoff_candidates, reward_gain, used)
    used.add(tradeoff)
    selected.append(
        (
            "Компромисс reward–relevance",
            "PPO повышает оценку RM, но вытесняет реальный следующий item из Top-5.",
            "По Recall это ошибка, однако сам список не обязательно плох: единственный "
            "наблюдавшийся следующий просмотр не является полным перечнем объектов, которые "
            "могли заинтересовать пользователя.",
            tradeoff,
        )
    )

    feedback_difference = np.asarray(
        [5 - _overlap(left, right) for left, right in zip(slate_rec, feedback_rec, strict=True)]
    )
    feedback_index = _choose_index(
        np.flatnonzero(feedback_difference > 0).tolist(),
        feedback_difference,
        used,
    )
    used.add(feedback_index)
    selected.append(
        (
            "Где slate и feedback расходятся",
            "Редкий пример влияния смоделированного feedback на последующие позиции.",
            "Здесь полезно оценивать не только совпадение с target, но и связность списка с "
            "историей. Пользовательские последовательности могут резко менять тему, поэтому "
            "семантически логичное продолжение иногда получает нулевой Recall.",
            feedback_index,
        )
    )

    common_slate, common_count = Counter(map(tuple, low_rec)).most_common(1)[0]
    collapse_candidates = [
        index
        for index, recommendations in enumerate(low_rec)
        if tuple(recommendations) == common_slate and base_rec[index] != recommendations
    ]
    collapse_score = np.asarray(
        [float(tuple(recommendations) == common_slate) for recommendations in low_rec]
    )
    collapse_index = _choose_index(collapse_candidates, collapse_score, used)
    selected.append(
        (
            "Схлопывание при β=0.05",
            f"Самый частый Top-5 слабого KL встретился {common_count} раз "
            f"в подвыборке из {len(low_rec)} историй.",
            "Для одного пользователя даже такой список может выглядеть уместно. Проблема "
            "обнаруживается на уровне всей выборки: одинаковый набор повторяется для разных "
            "историй, поэтому падают coverage и доля уникальных списков.",
            collapse_index,
        )
    )
    return selected


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _policy_summary(result: dict[str, Any]) -> dict[str, float]:
    recommendations = result["recommendations"]
    flat = [item for row in recommendations for item in row]
    counts = Counter(flat)
    targets = result["targets"]
    return {
        "recall": float(
            np.mean(
                [target in row for target, row in zip(targets, recommendations, strict=True)]
            )
        ),
        "reward": float(np.mean(result["mean_rewards"])),
        "unique_slates": len(set(map(tuple, recommendations))) / len(recommendations),
        "top10_share": sum(count for _, count in counts.most_common(10)) / len(flat),
    }


def _render_markdown(
    *,
    results: dict[str, dict[str, Any]],
    selected: list[tuple[str, str, str, int]],
    catalog: dict[int, dict[str, Any]],
    max_examples: int,
) -> str:
    labels = {spec.key: spec.label for spec in POLICIES}
    lines = [
        "# Примеры рекомендаций GPTRec и PPO",
        "",
        "Этот отчёт показывает отдельные test-истории, выбранные для иллюстрации уже",
        "измеренных на полном test закономерностей. Это не дополнительная оценка качества:",
        "примеры намеренно подобраны как понятные случаи, а общие выводы должны опираться",
        "на агрегированные метрики.",
        "",
        "Важно: test содержит один фактически просмотренный следующий item. Recall и NDCG",
        "считают остальные рекомендации нерелевантными, хотя часть из них может быть",
        "логичным и полезным продолжением истории. Поэтому ниже отдельно обсуждается как",
        "формальное попадание в target, так и содержательная правдоподобность списка.",
        "",
        "Пользовательские идентификаторы не сохраняются. История сокращена до последних",
        "пяти известных объектов.",
        "",
        "## Протокол",
        "",
        f"- равномерная детерминированная подвыборка: `{max_examples}` test-историй;",
        "- единый Top-100 от замороженной GPTRec-base;",
        "- последовательная greedy-генерация Top-5;",
        "- один ансамбль Reward Model и одна формула raw reward;",
        "- сравнение GPTRec-base, ранних PPO `β=0.5` и длинного PPO `β=0.05`.",
        "",
        "В таблицах примеров `C` — прогноз полноты просмотра, `D` — полезность",
        "длительности, `P(return)` — вероятность возвращения в сервис, а `U` —",
        "неуверенность ансамбля. Все четыре значения усреднены по пяти позициям Top-5.",
        "",
        "## Сводка по подвыборке",
        "",
        "| Policy | Recall@5 | Средний raw reward | Unique Top-5 | Top-10 share |",
        "|---|---:|---:|---:|---:|",
    ]
    for spec in POLICIES:
        summary = _policy_summary(results[spec.key])
        lines.append(
            f"| {spec.label} | {summary['recall']:.3f} | {summary['reward']:.3f} | "
            f"{summary['unique_slates']:.1%} | {summary['top10_share']:.1%} |"
        )
    slate_rows = results["slate_beta_05"]["recommendations"]
    feedback_rows = results["feedback_beta_05"]["recommendations"]
    exact_match_rate = np.mean(
        [left == right for left, right in zip(slate_rows, feedback_rows, strict=True)]
    )
    mean_overlap = np.mean(
        [
            _overlap(left, right) / len(left)
            for left, right in zip(slate_rows, feedback_rows, strict=True)
        ]
    )
    lines.extend(
        [
            "",
            f"На этой подвыборке Top-5 slate и feedback полностью совпали в "
            f"**{exact_match_rate:.1%}** случаев; среднее пересечение составило "
            f"**{mean_overlap:.1%}** items.",
        ]
    )

    for number, (title, explanation, interpretation, index) in enumerate(
        selected, start=1
    ):
        target = int(results["base"]["targets"][index])
        history = results["base"]["histories"][index]
        history_text = " → ".join(_item_text(token, catalog) for token in history[-5:])
        lines.extend(
            [
                "",
                f"## Пример {number}. {title}",
                "",
                explanation,
                "",
                f"**Последние объекты истории:** {_escape_markdown(history_text)}",
                "",
                f"**Реальный следующий item:** {_escape_markdown(_item_text(target, catalog))}",
                "",
                "| Policy | Top-5 рекомендаций | Target | Reward | C | D | P(return) | U |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for spec in POLICIES:
            result = results[spec.key]
            recommendations = result["recommendations"][index]
            recommendation_text = "<br>".join(
                f"{position}. {_escape_markdown(_item_text(token, catalog))}"
                for position, token in enumerate(recommendations, start=1)
            )
            rank = _rank(target, recommendations)
            target_text = f"#{rank}" if rank is not None else "—"
            lines.append(
                f"| {labels[spec.key]} | {recommendation_text} | {target_text} | "
                f"{result['mean_rewards'][index]:.3f} | "
                f"{result['mean_components']['completion_utility'][index]:.3f} | "
                f"{result['mean_components']['duration_utility'][index]:.3f} | "
                f"{result['mean_components']['return_probability'][index]:.3f} | "
                f"{result['mean_components']['epistemic_uncertainty'][index]:.3f} |"
            )
        lines.extend(["", f"**Интерпретация:** {interpretation}"])

    lines.extend(
        [
            "",
            "## Как интерпретировать",
            "",
            "- Ранний PPO с `β=0.5` обычно сохраняет структуру GPTRec-base, но переставляет",
            "  или заменяет часть items в пользу более высокого прогноза RM.",
            "- `PPO-slate` и `PPO-feedback` после 50 updates часто совпадают; отдельный",
            "  пример расхождения показывает возможное влияние feedback, но не доказывает",
            "  его систематическое преимущество.",
            "- Длинный запуск с `β=0.05` на разных историях часто возвращает одинаковые",
            "  списки. Это качественная иллюстрация падения coverage и unique slate rate.",
            "- Raw reward рассчитан той же RM, которую оптимизирует PPO, поэтому его рост",
            "  нельзя считать независимым доказательством пользовательской пользы.",
            "",
            "## Воспроизведение",
            "",
            "```bash",
            "uv run python scripts/generate_recommendation_examples.py",
            "```",
            "",
            "Для более быстрой проверки можно уменьшить подвыборку:",
            "",
            "```bash",
            "uv run python scripts/generate_recommendation_examples.py --max-examples 256",
            "```",
            "",
            "Скрипт перезаписывает этот Markdown и машинный JSON-отчёт в",
            "`reports/generated/recommendation_examples/examples.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_ready(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        key: {
            field: value
            for field, value in result.items()
            if field not in {"histories"}
        }
        for key, result in results.items()
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_ppo_high_kl.yaml")
    )
    parser.add_argument(
        "--items", type=Path, default=Path("data/interim/kion_common/items.parquet")
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=Path("data/processed/gptrec_base/item_vocabulary.parquet"),
    )
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/generated/recommendation_examples/examples.json"),
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("docs/recommendation-examples.md")
    )
    args = parser.parse_args()
    if args.max_examples < 4 or args.batch_size < 1:
        raise ValueError("max-examples must be at least 4 and batch-size must be positive")

    config = PPOTrainingConfig.from_yaml(args.config)
    device = _choose_device()
    torch.set_float32_matmul_precision("high")
    store = RewardEventStore.from_parquet(config.events_path, config.item_features_path)
    dataset = KIONRewardDataset(store, split="test")
    sample_count = min(len(dataset), args.max_examples)
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64).tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    reference = _load_base(config.base_checkpoint, device)
    reference.eval()
    reward_model = RewardModelEnsemble.from_checkpoints(
        _reward_checkpoints(config.reward_checkpoint_dir),
        backbone_checkpoint=config.base_checkpoint,
        map_location=device,
    ).to(device)
    reward_model.eval()
    feature_lookup = ItemFeatureLookup.from_parquet(
        config.item_features_path,
        vocab_size=reference.project_config.vocab_size,
        device=device,
    )
    catalog = _item_catalog(args.items, args.vocabulary)

    histories: list[list[int]] = []
    targets: list[int] = []
    for raw_batch in loader:
        for row, mask in zip(
            raw_batch["input_ids"].tolist(),
            raw_batch["attention_mask"].tolist(),
            strict=True,
        ):
            histories.append([token for token, keep in zip(row, mask, strict=True) if keep])
        targets.extend(raw_batch["action_id"].tolist())

    results: dict[str, dict[str, Any]] = {}
    for spec in POLICIES:
        print(f"start policy={spec.key} examples={sample_count} device={device.type}", flush=True)
        policy = _load_policy(
            spec,
            base_checkpoint=config.base_checkpoint,
            device=device,
        )
        policy.eval()
        batches: list[dict[str, Tensor]] = []
        for batch_index, raw_batch in enumerate(loader, start=1):
            batches.append(
                _generate_batch(
                    policy,
                    reference,
                    reward_model,
                    feature_lookup,
                    raw_batch,
                    config,
                    mode=spec.mode,
                    device=device,
                )
            )
            if batch_index % 10 == 0:
                print(f"policy={spec.key} batches={batch_index}", flush=True)
        combined = {
            name: torch.cat([batch[name] for batch in batches], dim=0)
            for name in ("recommendations", "rewards", *COMPONENTS)
        }
        results[spec.key] = {
            "label": spec.label,
            "mode": spec.mode,
            "checkpoint": str(spec.checkpoint or config.base_checkpoint),
            "histories": histories,
            "targets": targets,
            "recommendations": combined["recommendations"].tolist(),
            "mean_rewards": combined["rewards"].mean(dim=1).tolist(),
            "mean_components": {
                name: combined[name].mean(dim=1).tolist() for name in COMPONENTS
            },
        }
        del policy

    selected = _select_examples(results)
    report = {
        "protocol": {
            "split": "test",
            "sample_selection": "evenly spaced deterministic subset",
            "examples": sample_count,
            "candidate_count": config.candidate_count,
            "horizon": config.horizon,
            "generation": "greedy autoregressive Top-5",
        },
        "selected": [
            {
                "title": title,
                "description": description,
                "interpretation": interpretation,
                "sample_index": index,
            }
            for title, description, interpretation, index in selected
        ],
        "policies": _json_ready(results),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        _render_markdown(
            results=results,
            selected=selected,
            catalog=catalog,
            max_examples=sample_count,
        ),
        encoding="utf-8",
    )
    print(f"saved report={args.report}", flush=True)
    print(f"saved markdown={args.markdown}", flush=True)


if __name__ == "__main__":
    main()
