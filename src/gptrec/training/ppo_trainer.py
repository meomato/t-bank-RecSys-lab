"""KL-regularized PPO training for GPTRec slate and feedback policies."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from gptrec.data.reward_dataset import KIONRewardDataset, RewardEventStore
from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig
from gptrec.models.ppo_policy import (
    GPTRecActorCritic,
    PPOPolicyConfig,
    append_policy_step,
)
from gptrec.models.reward_ensemble import RewardModelEnsemble

RolloutMode = Literal["slate", "feedback"]
REWARD_COMPONENTS = (
    "completion_utility",
    "duration_utility",
    "return_probability",
    "epistemic_uncertainty",
)


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    events_path: Path
    item_features_path: Path
    base_checkpoint: Path
    reward_checkpoint_dir: Path
    checkpoint_root: Path
    report_root: Path
    modes: tuple[RolloutMode, ...]
    seed: int
    updates: int
    rollout_batch_size: int
    candidate_count: int
    horizon: int
    ppo_epochs: int
    minibatch_size: int
    learning_rate: float
    weight_decay: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    kl_beta: float
    max_grad_norm: float
    normalization_batches: int
    eval_every_updates: int
    eval_batches: int
    test_batches: int
    minimum_ndcg_ratio: float
    num_workers: int
    reward_weights: tuple[float, float, float, float]

    @classmethod
    def from_yaml(cls, path: str | Path) -> PPOTrainingConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        paths = document["paths"]
        training = document["training"]
        reward = document.get("reward", {})
        modes = tuple(training.get("modes", ("slate", "feedback")))
        if any(mode not in ("slate", "feedback") for mode in modes):
            raise ValueError("PPO modes must be slate and/or feedback")
        config = cls(
            events_path=Path(paths["events"]),
            item_features_path=Path(paths["item_features"]),
            base_checkpoint=Path(paths["base_checkpoint"]),
            reward_checkpoint_dir=Path(paths["reward_checkpoint_dir"]),
            checkpoint_root=Path(paths["checkpoint_root"]),
            report_root=Path(paths["report_root"]),
            modes=modes,
            seed=int(training["seed"]),
            updates=int(training["updates"]),
            rollout_batch_size=int(training["rollout_batch_size"]),
            candidate_count=int(training["candidate_count"]),
            horizon=int(training["horizon"]),
            ppo_epochs=int(training["ppo_epochs"]),
            minibatch_size=int(training["minibatch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            gamma=float(training["gamma"]),
            gae_lambda=float(training["gae_lambda"]),
            clip_epsilon=float(training["clip_epsilon"]),
            value_coefficient=float(training["value_coefficient"]),
            entropy_coefficient=float(training["entropy_coefficient"]),
            kl_beta=float(training["kl_beta"]),
            max_grad_norm=float(training["max_grad_norm"]),
            normalization_batches=int(training["normalization_batches"]),
            eval_every_updates=int(training["eval_every_updates"]),
            eval_batches=int(training["eval_batches"]),
            test_batches=int(training.get("test_batches", 0)),
            minimum_ndcg_ratio=float(training.get("minimum_ndcg_ratio", 0.90)),
            num_workers=int(training.get("num_workers", 0)),
            reward_weights=(
                float(reward.get("completion_weight", 0.55)),
                float(reward.get("duration_weight", 0.30)),
                float(reward.get("return_weight", 0.15)),
                float(reward.get("uncertainty_weight", 0.25)),
            ),
        )
        if not config.modes or min(
            config.updates,
            config.rollout_batch_size,
            config.candidate_count,
            config.horizon,
            config.ppo_epochs,
            config.minibatch_size,
            config.normalization_batches,
            config.eval_every_updates,
            config.eval_batches,
        ) < 1:
            raise ValueError("PPO sizes and counts must be positive")
        if config.test_batches < 0:
            raise ValueError("test_batches must be non-negative; zero means all batches")
        if not 0 <= config.gamma <= 1 or not 0 <= config.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if not 0 <= config.minimum_ndcg_ratio <= 1:
            raise ValueError("minimum_ndcg_ratio must be in [0, 1]")
        return config

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "events_path",
            "item_features_path",
            "base_checkpoint",
            "reward_checkpoint_dir",
            "checkpoint_root",
            "report_root",
        ):
            result[key] = str(result[key])
        result["modes"] = list(self.modes)
        result["reward_weights"] = list(self.reward_weights)
        return result


@dataclass(frozen=True, slots=True)
class RewardNormalizer:
    """Frozen component-wise z-score statistics from base-policy rollouts."""

    means: dict[str, float]
    standard_deviations: dict[str, float]
    clip: float = 5.0

    @classmethod
    def fit(cls, samples: dict[str, Tensor], *, minimum_std: float = 1e-4) -> RewardNormalizer:
        means: dict[str, float] = {}
        deviations: dict[str, float] = {}
        for name in REWARD_COMPONENTS:
            values = samples[name].float().reshape(-1)
            if not values.numel():
                raise ValueError(f"cannot normalize empty reward component {name}")
            means[name] = float(values.mean())
            deviations[name] = max(float(values.std(unbiased=False)), minimum_std)
        return cls(means, deviations)

    def transform(self, predictions: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            name: (
                (predictions[name] - self.means[name]) / self.standard_deviations[name]
            ).clamp(-self.clip, self.clip)
            for name in REWARD_COMPONENTS
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "means": self.means,
            "standard_deviations": self.standard_deviations,
            "clip": self.clip,
        }


@dataclass(slots=True)
class ItemFeatureLookup:
    content_type_ids: Tensor
    numeric_features: Tensor

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        vocab_size: int,
        device: torch.device,
    ) -> ItemFeatureLookup:
        frame = pd.read_parquet(path)
        numeric_columns = ["release_year_scaled", "log_train_popularity"] + sorted(
            column for column in frame if column.startswith("genre_")
        )
        content_types = torch.zeros(vocab_size, dtype=torch.long)
        numeric = torch.zeros(vocab_size, len(numeric_columns), dtype=torch.float32)
        tokens = torch.from_numpy(frame["item_token_id"].to_numpy(dtype=np.int64))
        content_types[tokens] = torch.from_numpy(
            frame["content_type_id"].to_numpy(dtype=np.int64)
        )
        numeric[tokens] = torch.from_numpy(
            frame[numeric_columns].to_numpy(dtype=np.float32)
        )
        return cls(content_types.to(device), numeric.to(device))

    def for_actions(self, action_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.content_type_ids[action_ids], self.numeric_features[action_ids]


@dataclass(slots=True)
class RolloutBatch:
    input_ids: Tensor
    attention_mask: Tensor
    feedback_features: Tensor
    candidate_ids: Tensor
    action_indices: Tensor
    old_log_probabilities: Tensor
    reference_log_probabilities: Tensor
    returns: Tensor
    advantages: Tensor
    rewards: Tensor
    raw_rewards: Tensor

    def flattened(self) -> dict[str, Tensor]:
        batch_size, horizon = self.action_indices.shape
        count = batch_size * horizon
        return {
            "input_ids": self.input_ids.reshape(count, -1),
            "attention_mask": self.attention_mask.reshape(count, -1),
            "feedback_features": self.feedback_features.reshape(
                count, self.feedback_features.size(-2), self.feedback_features.size(-1)
            ),
            "candidate_ids": self.candidate_ids.reshape(count, -1),
            "action_indices": self.action_indices.reshape(count),
            "old_log_probabilities": self.old_log_probabilities.reshape(count),
            "reference_log_probabilities": self.reference_log_probabilities.reshape(
                count, -1
            ),
            "returns": self.returns.reshape(count),
            "advantages": self.advantages.reshape(count),
        }


def compose_normalized_reward(
    predictions: dict[str, Tensor],
    normalizer: RewardNormalizer,
    weights: tuple[float, float, float, float],
) -> Tensor:
    normalized = normalizer.transform(predictions)
    completion, duration, return_weight, uncertainty = weights
    return (
        completion * normalized["completion_utility"]
        + duration * normalized["duration_utility"]
        + return_weight * normalized["return_probability"]
        - uncertainty * normalized["epistemic_uncertainty"]
    )


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute finite-horizon GAE with zero terminal bootstrap value."""
    if rewards.shape != values.shape or rewards.ndim != 2:
        raise ValueError("rewards and values must have shape [batch, horizon]")
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(rewards.size(0), device=rewards.device)
    next_value = torch.zeros(rewards.size(0), device=rewards.device)
    for step in range(rewards.size(1) - 1, -1, -1):
        delta = rewards[:, step] + gamma * next_value - values[:, step]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[:, step] = next_advantage
        next_value = values[:, step]
    return advantages, advantages + values


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_base(path: Path, device: torch.device) -> GPTRecBase:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GPTRecBase(GPTRecBaseConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def _reward_checkpoints(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("seed_*/best.pt"))
    if not paths:
        raise FileNotFoundError(f"no reward checkpoints found under {directory}")
    return paths


def _evenly_spaced_subset(
    dataset: Dataset[dict[str, Tensor]],
    *,
    batch_size: int,
    batches: int,
) -> Dataset[dict[str, Tensor]]:
    """Bound evaluation cost without taking only the first users in storage order."""
    if batches == 0:
        return dataset
    sample_count = min(len(dataset), batch_size * batches)
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)
    return Subset(dataset, indices.tolist())


def _candidate_ids(reference_logits: Tensor, input_ids: Tensor, count: int) -> Tensor:
    ranking = reference_logits.clone()
    ranking[:, 0] = -torch.inf
    ranking.scatter_(1, input_ids, -torch.inf)
    available = ranking.size(1) - 1
    if count > available:
        raise ValueError(f"candidate_count={count} exceeds catalogue size {available}")
    return ranking.topk(count, dim=-1).indices


def _policy_distribution(logits: Tensor, candidate_ids: Tensor) -> tuple[Tensor, Tensor]:
    candidate_logits = logits.gather(1, candidate_ids)
    log_probabilities = candidate_logits.log_softmax(dim=-1)
    return log_probabilities, log_probabilities.exp()


def _raw_reward(
    predictions: dict[str, Tensor],
    weights: tuple[float, float, float, float],
) -> Tensor:
    completion, duration, return_weight, uncertainty = weights
    return (
        completion * predictions["completion_utility"]
        + duration * predictions["duration_utility"]
        + return_weight * predictions["return_probability"]
        - uncertainty * predictions["epistemic_uncertainty"]
    )


def _feedback_vector(predictions: dict[str, Tensor], *, mode: RolloutMode) -> Tensor:
    if mode == "slate":
        return torch.zeros(
            predictions["completion_utility"].size(0),
            3,
            device=predictions["completion_utility"].device,
        )
    return torch.stack(
        [
            predictions["completion_utility"],
            predictions["duration_utility"],
            predictions["return_probability"],
        ],
        dim=-1,
    )


@torch.no_grad()
def fit_base_reward_normalizer(
    reference: GPTRecBase,
    reward_model: RewardModelEnsemble,
    feature_lookup: ItemFeatureLookup,
    loader: DataLoader,
    *,
    device: torch.device,
    candidate_count: int,
    batches: int,
) -> RewardNormalizer:
    """Estimate fixed reward scales from reference-policy actions."""
    collected = {name: [] for name in REWARD_COMPONENTS}
    reference.eval()
    reward_model.eval()
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= batches:
            break
        input_ids = raw_batch["input_ids"].to(device)
        attention_mask = raw_batch["attention_mask"].to(device)
        logits = reference.next_item_logits(input_ids, attention_mask)
        candidates = _candidate_ids(logits, input_ids, candidate_count)
        candidate_log_probabilities, _ = _policy_distribution(logits, candidates)
        action_indices = torch.distributions.Categorical(
            logits=candidate_log_probabilities
        ).sample()
        actions = candidates.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        content_types, numeric = feature_lookup.for_actions(actions)
        predictions = reward_model.predict(
            input_ids,
            attention_mask,
            actions,
            content_types,
            numeric,
        )
        for name in REWARD_COMPONENTS:
            collected[name].append(predictions[name].cpu())
    return RewardNormalizer.fit({name: torch.cat(values) for name, values in collected.items()})


@torch.no_grad()
def collect_rollout(
    policy: GPTRecActorCritic,
    reference: GPTRecBase,
    reward_model: RewardModelEnsemble,
    feature_lookup: ItemFeatureLookup,
    raw_batch: dict[str, Tensor],
    normalizer: RewardNormalizer,
    *,
    mode: RolloutMode,
    candidate_count: int,
    horizon: int,
    gamma: float,
    gae_lambda: float,
    reward_weights: tuple[float, float, float, float],
    device: torch.device,
) -> RolloutBatch:
    policy.eval()
    reference.eval()
    input_ids = raw_batch["input_ids"].to(device)
    attention_mask = raw_batch["attention_mask"].to(device)
    feedback = torch.zeros(
        *input_ids.shape,
        policy.config.feedback_feature_dim,
        device=device,
    )
    states: dict[str, list[Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "feedback_features": [],
        "candidate_ids": [],
        "action_indices": [],
        "old_log_probabilities": [],
        "reference_log_probabilities": [],
        "values": [],
        "rewards": [],
        "raw_rewards": [],
    }
    for _ in range(horizon):
        actor_logits, values = policy(input_ids, attention_mask, feedback)
        reference_logits = reference.next_item_logits(input_ids, attention_mask)
        candidates = _candidate_ids(reference_logits, input_ids, candidate_count)
        actor_log_probabilities, _ = _policy_distribution(actor_logits, candidates)
        reference_log_probabilities, _ = _policy_distribution(reference_logits, candidates)
        action_indices = torch.distributions.Categorical(
            logits=actor_log_probabilities
        ).sample()
        actions = candidates.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        old_log_probability = actor_log_probabilities.gather(
            1, action_indices.unsqueeze(1)
        ).squeeze(1)
        content_types, numeric = feature_lookup.for_actions(actions)
        predictions = reward_model.predict(
            input_ids,
            attention_mask,
            actions,
            content_types,
            numeric,
        )
        rewards = compose_normalized_reward(predictions, normalizer, reward_weights)

        states["input_ids"].append(input_ids.clone())
        states["attention_mask"].append(attention_mask.clone())
        states["feedback_features"].append(feedback.clone())
        states["candidate_ids"].append(candidates)
        states["action_indices"].append(action_indices)
        states["old_log_probabilities"].append(old_log_probability)
        states["reference_log_probabilities"].append(reference_log_probabilities)
        states["values"].append(values)
        states["rewards"].append(rewards)
        states["raw_rewards"].append(_raw_reward(predictions, reward_weights))

        input_ids, attention_mask, feedback = append_policy_step(
            input_ids,
            attention_mask,
            feedback,
            actions,
            _feedback_vector(predictions, mode=mode),
        )

    values = torch.stack(states["values"], dim=1)
    rewards = torch.stack(states["rewards"], dim=1)
    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(
        1e-6
    )
    return RolloutBatch(
        input_ids=torch.stack(states["input_ids"], dim=1),
        attention_mask=torch.stack(states["attention_mask"], dim=1),
        feedback_features=torch.stack(states["feedback_features"], dim=1),
        candidate_ids=torch.stack(states["candidate_ids"], dim=1),
        action_indices=torch.stack(states["action_indices"], dim=1),
        old_log_probabilities=torch.stack(states["old_log_probabilities"], dim=1),
        reference_log_probabilities=torch.stack(
            states["reference_log_probabilities"], dim=1
        ),
        returns=returns,
        advantages=advantages,
        rewards=rewards,
        raw_rewards=torch.stack(states["raw_rewards"], dim=1),
    )


def ppo_update(
    policy: GPTRecActorCritic,
    optimizer: AdamW,
    rollout: RolloutBatch,
    config: PPOTrainingConfig,
) -> dict[str, float]:
    policy.train()
    data = rollout.flattened()
    count = data["action_indices"].numel()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    updates = 0
    for _ in range(config.ppo_epochs):
        for indices in torch.randperm(count, device=data["action_indices"].device).split(
            config.minibatch_size
        ):
            logits, values = policy(
                data["input_ids"][indices],
                data["attention_mask"][indices],
                data["feedback_features"][indices],
            )
            log_probabilities, probabilities = _policy_distribution(
                logits, data["candidate_ids"][indices]
            )
            chosen = log_probabilities.gather(
                1, data["action_indices"][indices].unsqueeze(1)
            ).squeeze(1)
            ratios = (chosen - data["old_log_probabilities"][indices]).exp()
            advantages = data["advantages"][indices]
            unclipped = ratios * advantages
            clipped = ratios.clamp(
                1 - config.clip_epsilon, 1 + config.clip_epsilon
            ) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, data["returns"][indices])
            entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
            reference_log_probabilities = data["reference_log_probabilities"][indices]
            kl = (
                probabilities * (log_probabilities - reference_log_probabilities)
            ).sum(dim=-1).mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
                + config.kl_beta * kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
            for name, value in (
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("entropy", entropy),
                ("kl", kl),
            ):
                totals[name] += float(value.detach().cpu())
            updates += 1
    return {name: value / updates for name, value in totals.items()}


@torch.no_grad()
def evaluate_policy(
    policy: GPTRecActorCritic,
    reference: GPTRecBase,
    reward_model: RewardModelEnsemble,
    feature_lookup: ItemFeatureLookup,
    loader: DataLoader,
    normalizer: RewardNormalizer,
    config: PPOTrainingConfig,
    *,
    mode: RolloutMode,
    device: torch.device,
    max_batches: int | None = None,
    progress_label: str | None = None,
    progress_every_batches: int = 500,
) -> dict[str, float]:
    policy.eval()
    total = 0
    hit_sum = 0.0
    ndcg_sum = 0.0
    reward_sum = 0.0
    normalized_reward_sum = 0.0
    kl_sum = 0.0
    entropy_sum = 0.0
    popularity_sum = 0.0
    diversity_sum = 0.0
    duplicate_pair_sum = 0.0
    actions_seen: set[int] = set()
    action_counts: Counter[int] = Counter()
    slates_seen: set[tuple[int, ...]] = set()
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = raw_batch["input_ids"].to(device)
        attention_mask = raw_batch["attention_mask"].to(device)
        targets = raw_batch["action_id"].to(device)
        feedback = torch.zeros(
            *input_ids.shape,
            policy.config.feedback_feature_dim,
            device=device,
        )
        generated: list[Tensor] = []
        for _ in range(config.horizon):
            actor_logits, _ = policy(input_ids, attention_mask, feedback)
            reference_logits = reference.next_item_logits(input_ids, attention_mask)
            candidates = _candidate_ids(reference_logits, input_ids, config.candidate_count)
            actor_log_probabilities, probabilities = _policy_distribution(
                actor_logits, candidates
            )
            reference_log_probabilities, _ = _policy_distribution(
                reference_logits, candidates
            )
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
            reward_sum += float(_raw_reward(predictions, config.reward_weights).sum().cpu())
            normalized_reward_sum += float(
                compose_normalized_reward(predictions, normalizer, config.reward_weights)
                .sum()
                .cpu()
            )
            kl_sum += float(
                (
                    probabilities
                    * (actor_log_probabilities - reference_log_probabilities)
                )
                .sum()
                .cpu()
            )
            entropy_sum += float(
                (-(probabilities * actor_log_probabilities).sum(dim=-1)).sum().cpu()
            )
            generated.append(actions)
            actions_seen.update(actions.cpu().tolist())
            input_ids, attention_mask, feedback = append_policy_step(
                input_ids,
                attention_mask,
                feedback,
                actions,
                _feedback_vector(predictions, mode=mode),
            )
        recommendations = torch.stack(generated, dim=1)
        recommendation_rows = recommendations.cpu().tolist()
        action_counts.update(item for row in recommendation_rows for item in row)
        slates_seen.update(tuple(row) for row in recommendation_rows)
        recommendation_features = feature_lookup.numeric_features[recommendations]
        popularity_sum += float(recommendation_features[:, :, 1].sum().cpu())
        if config.horizon > 1:
            genres = F.normalize(recommendation_features[:, :, 2:], dim=-1, eps=1e-8)
            similarities = torch.bmm(genres, genres.transpose(1, 2))
            pair_mask = torch.triu(
                torch.ones(
                    config.horizon,
                    config.horizon,
                    dtype=torch.bool,
                    device=device,
                ),
                diagonal=1,
            )
            pair_count = pair_mask.sum()
            diversity_sum += float(
                (1 - similarities[:, pair_mask]).sum().cpu() / pair_count
            )
            duplicate_pair_sum += float(
                recommendations.unsqueeze(2)
                .eq(recommendations.unsqueeze(1))[:, pair_mask]
                .float()
                .sum()
                .cpu()
                / pair_count
            )
        hits = recommendations.eq(targets.unsqueeze(1))
        hit_sum += float(hits.any(dim=1).sum().cpu())
        discounts = torch.log2(
            torch.arange(2, config.horizon + 2, device=device, dtype=torch.float32)
        )
        ndcg_sum += float((hits.float() / discounts).sum().cpu())
        total += targets.size(0)
        if (
            progress_label is not None
            and progress_every_batches > 0
            and (batch_index + 1) % progress_every_batches == 0
        ):
            print(
                f"evaluation policy={progress_label} batches={batch_index + 1} "
                f"examples={total:,}",
                flush=True,
            )
    if not total:
        raise ValueError("evaluation produced no examples")
    action_count = total * config.horizon
    frequency_ranking = sorted(action_counts.values(), reverse=True)
    return {
        f"recall@{config.horizon}": hit_sum / total,
        f"ndcg@{config.horizon}": ndcg_sum / total,
        "raw_reward_per_step": reward_sum / action_count,
        "normalized_reward_per_step": normalized_reward_sum / action_count,
        "kl_per_step": kl_sum / action_count,
        "entropy_per_step": entropy_sum / action_count,
        "mean_log_train_popularity": popularity_sum / action_count,
        "intra_list_diversity": diversity_sum / total if config.horizon > 1 else 0.0,
        "duplicate_pair_rate": duplicate_pair_sum / total if config.horizon > 1 else 0.0,
        "catalog_coverage": len(actions_seen)
        / (feature_lookup.content_type_ids.numel() - 1),
        "top1_action_share": frequency_ranking[0] / action_count,
        "top10_action_share": sum(frequency_ranking[:10]) / action_count,
        "unique_slate_rate": len(slates_seen) / total,
        "examples": float(total),
    }


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _checkpoint_payload(
    policy: GPTRecActorCritic,
    optimizer: AdamW,
    config: PPOTrainingConfig,
    normalizer: RewardNormalizer,
    *,
    mode: RolloutMode,
    update: int,
) -> dict[str, Any]:
    return {
        **policy.policy_state_dict(),
        "mode": mode,
        "training_config": config.to_dict(),
        "reward_normalizer": normalizer.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "updates": update,
    }


def train_ppo_mode(
    config: PPOTrainingConfig,
    *,
    mode: RolloutMode,
    max_updates: int | None = None,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    device = _choose_device()
    store = RewardEventStore.from_parquet(config.events_path, config.item_features_path)
    train_dataset = KIONRewardDataset(store, split="train")
    validation_dataset = KIONRewardDataset(store, split="calibration")
    test_dataset = KIONRewardDataset(store, split="test")
    validation_dataset = _evenly_spaced_subset(
        validation_dataset,
        batch_size=config.rollout_batch_size,
        batches=config.eval_batches,
    )
    test_dataset = _evenly_spaced_subset(
        test_dataset,
        batch_size=config.rollout_batch_size,
        batches=config.test_batches,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.rollout_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.rollout_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.rollout_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    reference = _load_base(config.base_checkpoint, device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    policy = GPTRecActorCritic.from_gptrec_checkpoint(
        config.base_checkpoint,
        PPOPolicyConfig(),
        map_location=device,
    ).to(device)
    reward_model = RewardModelEnsemble.from_checkpoints(
        _reward_checkpoints(config.reward_checkpoint_dir),
        backbone_checkpoint=config.base_checkpoint,
        map_location=device,
    ).to(device)
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad = False
    feature_lookup = ItemFeatureLookup.from_parquet(
        config.item_features_path,
        vocab_size=reference.project_config.vocab_size,
        device=device,
    )
    normalizer = fit_base_reward_normalizer(
        reference,
        reward_model,
        feature_lookup,
        train_loader,
        device=device,
        candidate_count=config.candidate_count,
        batches=config.normalization_batches,
    )
    optimizer = AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_dir = config.checkpoint_root / mode
    report_dir = config.report_root / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    base_policy = GPTRecActorCritic.from_gptrec_checkpoint(
        config.base_checkpoint,
        PPOPolicyConfig(),
        map_location=device,
    ).to(device)
    base_validation = evaluate_policy(
        base_policy,
        reference,
        reward_model,
        feature_lookup,
        validation_loader,
        normalizer,
        config,
        mode=mode,
        device=device,
        max_batches=config.eval_batches,
    )
    del base_policy
    updates = min(config.updates, max_updates) if max_updates is not None else config.updates
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_is_eligible = False
    best_update = 0
    train_iterator = iter(train_loader)
    print(
        f"ppo mode={mode} device={device.type} updates={updates} "
        f"horizon={config.horizon} candidates={config.candidate_count}",
        flush=True,
    )
    for update in range(1, updates + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        rollout = collect_rollout(
            policy,
            reference,
            reward_model,
            feature_lookup,
            batch,
            normalizer,
            mode=mode,
            candidate_count=config.candidate_count,
            horizon=config.horizon,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            reward_weights=config.reward_weights,
            device=device,
        )
        metrics: dict[str, Any] = {
            "update": update,
            "rollout_reward": float(rollout.rewards.mean().cpu()),
            "rollout_raw_reward": float(rollout.raw_rewards.mean().cpu()),
        }
        metrics.update(ppo_update(policy, optimizer, rollout, config))
        if update % config.eval_every_updates == 0 or update == updates:
            validation = evaluate_policy(
                policy,
                reference,
                reward_model,
                feature_lookup,
                validation_loader,
                normalizer,
                config,
                mode=mode,
                device=device,
                max_batches=config.eval_batches,
            )
            metrics["validation"] = validation
            ndcg_name = f"ndcg@{config.horizon}"
            eligible = validation[ndcg_name] >= (
                config.minimum_ndcg_ratio * base_validation[ndcg_name]
            )
            score = (
                validation["normalized_reward_per_step"]
                - config.kl_beta * validation["kl_per_step"]
            )
            metrics["selection"] = {"eligible": eligible, "score": score}
            torch.save(
                _checkpoint_payload(
                    policy,
                    optimizer,
                    config,
                    normalizer,
                    mode=mode,
                    update=update,
                ),
                checkpoint_dir / f"update_{update:03d}.pt",
            )
            if (eligible and not best_is_eligible) or (
                eligible == best_is_eligible and score > best_score
            ):
                best_score = score
                best_is_eligible = eligible
                best_update = update
                torch.save(
                    _checkpoint_payload(
                        policy,
                        optimizer,
                        config,
                        normalizer,
                        mode=mode,
                        update=update,
                    ),
                    checkpoint_dir / "best.pt",
                )
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    checkpoint = _checkpoint_payload(
        policy,
        optimizer,
        config,
        normalizer,
        mode=mode,
        update=updates,
    )
    torch.save(checkpoint, checkpoint_dir / "final.pt")
    best_checkpoint = torch.load(
        checkpoint_dir / "best.pt", map_location=device, weights_only=False
    )
    policy.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate_policy(
        policy,
        reference,
        reward_model,
        feature_lookup,
        test_loader,
        normalizer,
        config,
        mode=mode,
        device=device,
        max_batches=config.test_batches or None,
    )
    result = {
        "mode": mode,
        "device": device.type,
        "config": config.to_dict(),
        "reward_normalizer": normalizer.to_dict(),
        "base_validation": base_validation,
        "best_update": best_update,
        "best_validation_score": best_score,
        "best_satisfies_ndcg_constraint": best_is_eligible,
        "test": test_metrics,
        "history": history,
    }
    _save_json(report_dir / "result.json", result)
    return result


def run_ppo_training(
    config: PPOTrainingConfig,
    *,
    mode: RolloutMode | None = None,
    max_updates: int | None = None,
) -> dict[str, Any]:
    modes = (mode,) if mode is not None else config.modes
    return {
        current_mode: train_ppo_mode(config, mode=current_mode, max_updates=max_updates)
        for current_mode in modes
    }
