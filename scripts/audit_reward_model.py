"""Audit a trained reward ensemble for popularity shortcuts and history neglect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gptrec.data.reward_dataset import KIONRewardDataset, RewardEventStore
from gptrec.models.reward_ensemble import RewardModelEnsemble
from gptrec.training.reward_trainer import RewardTrainingConfig


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_reward_model.yaml"))
    parser.add_argument("--max-examples", type=int, default=20_000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = RewardTrainingConfig.from_yaml(args.config)
    checkpoint_root = (
        Path("artifacts/checkpoints/reward_model_smoke")
        if args.smoke
        else config.checkpoint_dir
    )
    checkpoints = sorted(checkpoint_root.glob("seed_*/best.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no reward checkpoints under {checkpoint_root}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ensemble = RewardModelEnsemble.from_checkpoints(
        checkpoints,
        backbone_checkpoint=config.backbone_checkpoint,
        map_location=device,
    ).to(device)
    store = RewardEventStore.from_parquet(config.events_path, config.item_features_path)
    dataset = KIONRewardDataset(store, split="test", max_examples=args.max_examples)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    core_values: list[np.ndarray] = []
    shuffled_differences: list[np.ndarray] = []
    popularity_values: list[np.ndarray] = []
    uncertainty_values: list[np.ndarray] = []
    for batch in loader:
        model_inputs = {
            name: batch[name].to(device)
            for name in (
                "input_ids",
                "attention_mask",
                "action_id",
                "content_type_id",
                "item_numeric_features",
            )
        }
        prediction = ensemble.predict(**model_inputs)
        core = (
            0.55 * prediction["completion_utility"]
            + 0.30 * prediction["duration_utility"]
            + 0.15 * prediction["return_probability"]
        )
        shuffled_inputs = dict(model_inputs)
        shuffled_inputs["input_ids"] = model_inputs["input_ids"].roll(1, dims=0)
        shuffled_inputs["attention_mask"] = model_inputs["attention_mask"].roll(1, dims=0)
        shuffled = ensemble.predict(**shuffled_inputs)
        shuffled_core = (
            0.55 * shuffled["completion_utility"]
            + 0.30 * shuffled["duration_utility"]
            + 0.15 * shuffled["return_probability"]
        )
        core_values.append(core.cpu().numpy())
        shuffled_differences.append(torch.abs(core - shuffled_core).cpu().numpy())
        popularity_values.append(batch["item_numeric_features"][:, 1].numpy())
        uncertainty_values.append(prediction["epistemic_uncertainty"].cpu().numpy())
    core = np.concatenate(core_values)
    popularity = np.concatenate(popularity_values)
    uncertainty = np.concatenate(uncertainty_values)
    tail = popularity <= np.quantile(popularity, 0.1)
    head = popularity >= np.quantile(popularity, 0.9)
    result = {
        "examples": int(len(core)),
        "reward_popularity_spearman": _spearman(core, popularity),
        "mean_absolute_history_shuffle_effect": float(
            np.concatenate(shuffled_differences).mean()
        ),
        "tail_mean_reward": float(core[tail].mean()),
        "head_mean_reward": float(core[head].mean()),
        "tail_mean_uncertainty": float(uncertainty[tail].mean()),
        "head_mean_uncertainty": float(uncertainty[head].mean()),
        "warnings": [
            "A near-zero shuffle effect suggests that the model ignores user history.",
            "A large reward-popularity correlation requires matched popularity-slice checks.",
            "Smoke-model audit values are pipeline checks, not scientific results.",
        ],
    }
    output_dir = config.report_dir if not args.smoke else Path(
        "reports/generated/reward_model_smoke"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "shortcut_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
