"""Small validation sensitivity analysis for the scalar reward formula."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from gptrec.data.sequence_dataset import GPTRecEvalDataset, GPTRecEventStore
from gptrec.models.reward_ensemble import RewardModelEnsemble

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True, slots=True)
class RewardVariant:
    completion: float
    duration: float
    return_probability: float
    uncertainty: float


VARIANTS = {
    "base": RewardVariant(0.55, 0.30, 0.15, 0.25),
    "C−": RewardVariant(0.45, 0.3667, 0.1833, 0.25),
    "C+": RewardVariant(0.65, 0.2333, 0.1167, 0.25),
    "D−": RewardVariant(0.6286, 0.20, 0.1714, 0.25),
    "D+": RewardVariant(0.4714, 0.40, 0.1286, 0.25),
    "P−": RewardVariant(0.6147, 0.3353, 0.05, 0.25),
    "P+": RewardVariant(0.4853, 0.2647, 0.25, 0.25),
    "U=0": RewardVariant(0.55, 0.30, 0.15, 0.0),
    "U=0.10": RewardVariant(0.55, 0.30, 0.15, 0.10),
    "U=0.50": RewardVariant(0.55, 0.30, 0.15, 0.50),
}


def _reward(prediction: dict[str, Tensor], variant: RewardVariant) -> Tensor:
    return (
        variant.completion * prediction["completion_utility"]
        + variant.duration * prediction["duration_utility"]
        + variant.return_probability * prediction["return_probability"]
        - variant.uncertainty * prediction["epistemic_uncertainty"]
    )


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _item_feature_lookup(path: Path, vocab_size: int) -> tuple[Tensor, Tensor]:
    frame = pd.read_parquet(path)
    numeric_columns = ["release_year_scaled", "log_train_popularity"] + sorted(
        column for column in frame if column.startswith("genre_")
    )
    tokens = frame["item_token_id"].to_numpy(dtype=np.int64)
    content_type = np.zeros(vocab_size, dtype=np.int64)
    numeric = np.zeros((vocab_size, len(numeric_columns)), dtype=np.float32)
    valid = tokens < vocab_size
    content_type[tokens[valid]] = frame.loc[valid, "content_type_id"].to_numpy(
        dtype=np.int64
    )
    numeric[tokens[valid]] = frame.loc[valid, numeric_columns].to_numpy(dtype=np.float32)
    return torch.from_numpy(content_type), torch.from_numpy(numeric)


def _save_plot(result: dict[str, Any], output: Path) -> None:
    names = list(VARIANTS)
    metrics = result["variants"]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))

    overlap = [metrics[name]["top10_overlap_with_base_reward"] for name in names]
    colors = ["#16a34a" if name == "base" else "#64748b" for name in names]
    axes[0].barh(y, overlap, color=colors)
    axes[0].set(
        title="Устойчивость top-10 RM",
        xlabel="Overlap с базовой формулой",
        yticks=y,
        yticklabels=names,
        xlim=(0, 1.02),
    )
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.2)

    component_names = ("completion", "duration", "return", "uncertainty")
    base_values = np.array(
        [metrics["base"][f"predicted_{name}@10"] for name in component_names]
    )
    changes = np.array(
        [
            [metrics[name][f"predicted_{component}@10"] for component in component_names]
            for name in names
        ]
    ) - base_values
    scale = max(float(np.abs(changes).max()), 1e-6)
    image = axes[1].imshow(changes, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[1].set(
        title="Изменение состава top-10",
        xticks=np.arange(len(component_names)),
        xticklabels=("C", "D", "P(return)", "U"),
        yticks=y,
        yticklabels=names,
    )
    for row in range(len(names)):
        for column in range(len(component_names)):
            axes[1].text(
                column,
                row,
                f"{changes[row, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Δ predicted value")
    fig.suptitle(
        f"Sensitivity формулы reward · validation · {result['examples']:,} примеров · "
        f"{result['candidate_count']} кандидатов на контекст",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", type=Path, default=Path("data/processed/gptrec_base/events.parquet")
    )
    parser.add_argument(
        "--item-features",
        type=Path,
        default=Path("data/processed/reward_model/item_features.parquet"),
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=Path("artifacts/checkpoints/gptrec_base_standard/best.pt"),
    )
    parser.add_argument(
        "--checkpoints", type=Path, default=Path("artifacts/checkpoints/reward_model")
    )
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--max-examples", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/generated/reward_model/reward_sensitivity.json"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("docs/images/reward-model-sensitivity.png"),
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.candidate_count < args.cutoff:
        raise ValueError("candidate-count must be at least cutoff")
    if args.plot_only:
        with args.report.open(encoding="utf-8") as stream:
            result = json.load(stream)
        _save_plot(result, args.plot)
        print(f"saved plot={args.plot}", flush=True)
        return

    device = _choose_device()
    checkpoint_paths = sorted(args.checkpoints.glob("seed_*/best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"no reward checkpoints under {args.checkpoints}")
    print(f"loading ensemble members={len(checkpoint_paths)} device={device.type}", flush=True)
    ensemble = RewardModelEnsemble.from_checkpoints(
        checkpoint_paths,
        backbone_checkpoint=args.backbone,
        map_location=device,
    ).to(device)
    ensemble.eval()
    backbone = ensemble.models[0].backbone

    store = GPTRecEventStore.from_parquet(args.events)
    dataset = GPTRecEvalDataset(store, split="validation")
    example_count = min(len(dataset), args.max_examples)
    loader = DataLoader(
        Subset(dataset, range(example_count)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    vocab_size = backbone.project_config.vocab_size
    content_type, numeric_features = _item_feature_lookup(args.item_features, vocab_size)
    content_type = content_type.to(device)
    numeric_features = numeric_features.to(device)

    totals = {
        name: {
            "overlap": 0.0,
            "completion": 0.0,
            "duration": 0.0,
            "return": 0.0,
            "uncertainty": 0.0,
        }
        for name in VARIANTS
    }
    processed = 0
    top_k = args.cutoff
    for batch_index, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = backbone.next_item_logits(input_ids, attention_mask)
        logits[:, 0] = -torch.inf
        logits.scatter_(1, input_ids, -torch.inf)
        candidate_ids = logits.topk(args.candidate_count, dim=1).indices

        prediction = ensemble.predict_candidates(
            input_ids,
            attention_mask,
            candidate_ids,
            content_type[candidate_ids],
            numeric_features[candidate_ids],
        )
        rankings = {
            name: _reward(prediction, variant).argsort(dim=1, descending=True)
            for name, variant in VARIANTS.items()
        }
        base_top = rankings["base"][:, :top_k]
        for name, positions in rankings.items():
            top_positions = positions[:, :top_k]
            totals[name]["overlap"] += float(
                (top_positions.unsqueeze(2) == base_top.unsqueeze(1))
                .any(dim=2)
                .float()
                .sum()
                .cpu()
            )
            for component, key in (
                ("completion", "completion_utility"),
                ("duration", "duration_utility"),
                ("return", "return_probability"),
                ("uncertainty", "epistemic_uncertainty"),
            ):
                totals[name][component] += float(
                    prediction[key].gather(1, top_positions).sum().cpu()
                )

        processed += input_ids.size(0)
        if batch_index == 1 or batch_index % 20 == 0 or processed == example_count:
            print(
                f"sensitivity batch={batch_index}/{len(loader)} "
                f"examples={processed:,}/{example_count:,}",
                flush=True,
            )

    result: dict[str, Any] = {
        "split": "validation",
        "examples": processed,
        "candidate_count": args.candidate_count,
        "cutoff": top_k,
        "ensemble_members": len(checkpoint_paths),
        "uncertainty_definition": (
            "ensemble std of the base 0.55C + 0.30D + 0.15P(return) core"
        ),
        "variants": {},
    }
    for name, variant in VARIANTS.items():
        values = totals[name]
        result["variants"][name] = {
            "weights": asdict(variant),
            "top10_overlap_with_base_reward": values["overlap"] / (processed * top_k),
            **{
                f"predicted_{component}@10": values[component] / (processed * top_k)
                for component in ("completion", "duration", "return", "uncertainty")
            },
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    _save_plot(result, args.plot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved report={args.report} plot={args.plot}", flush=True)


if __name__ == "__main__":
    main()
