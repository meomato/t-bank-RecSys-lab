"""Evaluate all saved GPTRec/PPO policies with one full-test protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from gptrec.data.reward_dataset import KIONRewardDataset, RewardEventStore
from gptrec.models.ppo_policy import GPTRecActorCritic, PPOPolicyConfig
from gptrec.models.reward_ensemble import RewardModelEnsemble
from gptrec.training.ppo_trainer import (
    ItemFeatureLookup,
    PPOTrainingConfig,
    RewardNormalizer,
    _choose_device,
    _load_base,
    _reward_checkpoints,
    evaluate_policy,
)

RolloutMode = Literal["slate", "feedback"]


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    mode: RolloutMode
    checkpoint: Path | None


POLICIES = (
    PolicySpec("gptrec_base", "slate", None),
    PolicySpec(
        "ppo_slate_beta_005_best",
        "slate",
        Path("artifacts/checkpoints/ppo_final/slate/best.pt"),
    ),
    PolicySpec(
        "ppo_feedback_beta_005_best",
        "feedback",
        Path("artifacts/checkpoints/ppo_final/feedback/best.pt"),
    ),
    PolicySpec(
        "ppo_slate_beta_05_update_050",
        "slate",
        Path("artifacts/checkpoints/ppo_high_kl/slate/update_050.pt"),
    ),
    PolicySpec(
        "ppo_slate_beta_05_best",
        "slate",
        Path("artifacts/checkpoints/ppo_high_kl/slate/best.pt"),
    ),
    PolicySpec(
        "ppo_feedback_beta_05_update_050",
        "feedback",
        Path("artifacts/checkpoints/ppo_high_kl/feedback/update_050.pt"),
    ),
    PolicySpec(
        "ppo_feedback_beta_05_best",
        "feedback",
        Path("artifacts/checkpoints/ppo_high_kl/feedback/best.pt"),
    ),
)


def _load_normalizer(path: Path) -> RewardNormalizer:
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)["reward_normalizer"]
    return RewardNormalizer(
        means={name: float(value) for name, value in values["means"].items()},
        standard_deviations={
            name: float(value)
            for name, value in values["standard_deviations"].items()
        },
        clip=float(values["clip"]),
    )


def _load_policy(
    spec: PolicySpec,
    *,
    base_checkpoint: Path,
    device: torch.device,
) -> tuple[GPTRecActorCritic, dict[str, Any]]:
    if spec.checkpoint is None:
        policy = GPTRecActorCritic.from_gptrec_checkpoint(
            base_checkpoint,
            PPOPolicyConfig(),
            map_location=device,
        ).to(device)
        return policy, {"checkpoint": str(base_checkpoint), "update": 0, "beta": 0.0}

    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"missing policy checkpoint: {spec.checkpoint}")
    checkpoint = torch.load(spec.checkpoint, map_location=device, weights_only=False)
    policy_config = PPOPolicyConfig(**checkpoint.get("policy_config", {}))
    policy = GPTRecActorCritic.from_gptrec_checkpoint(
        base_checkpoint,
        policy_config,
        map_location=device,
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    training_config = checkpoint.get("training_config", {})
    return policy, {
        "checkpoint": str(spec.checkpoint),
        "update": int(checkpoint.get("updates", 0)),
        "beta": float(training_config.get("kl_beta", 0.0)),
    }


def _save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_ppo_high_kl.yaml"),
    )
    parser.add_argument(
        "--normalizer-report",
        type=Path,
        default=Path("reports/generated/ppo_high_kl/slate/result.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/ppo_comparison/full_test.json"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate policies already present in the output report.",
    )
    args = parser.parse_args()

    config = PPOTrainingConfig.from_yaml(args.config)
    normalizer = _load_normalizer(args.normalizer_report)
    device = _choose_device()
    torch.set_float32_matmul_precision("high")

    store = RewardEventStore.from_parquet(
        config.events_path,
        config.item_features_path,
    )
    test_dataset = KIONRewardDataset(store, split="test")
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

    if args.output.exists():
        with args.output.open(encoding="utf-8") as stream:
            report = json.load(stream)
    else:
        report = {
            "protocol": {
                "split": "test",
                "examples": len(test_dataset),
                "candidate_count": config.candidate_count,
                "horizon": config.horizon,
                "generation": "greedy autoregressive Top-5",
                "normalizer_report": str(args.normalizer_report),
            },
            "policies": {},
        }

    for spec in POLICIES:
        if spec.name in report["policies"] and not args.overwrite:
            print(f"skip policy={spec.name}: result already saved", flush=True)
            continue
        policy, metadata = _load_policy(
            spec,
            base_checkpoint=config.base_checkpoint,
            device=device,
        )
        print(
            f"start policy={spec.name} mode={spec.mode} device={device.type} "
            f"examples={len(test_dataset):,}",
            flush=True,
        )
        metrics = evaluate_policy(
            policy,
            reference,
            reward_model,
            feature_lookup,
            test_loader,
            normalizer,
            config,
            mode=spec.mode,
            device=device,
            max_batches=None,
            progress_label=spec.name,
        )
        report["policies"][spec.name] = {
            **metadata,
            "mode": spec.mode,
            "metrics": metrics,
        }
        _save(args.output, report)
        print(
            f"done policy={spec.name} recall@5={metrics['recall@5']:.6f} "
            f"ndcg@5={metrics['ndcg@5']:.6f} "
            f"reward={metrics['raw_reward_per_step']:.6f}",
            flush=True,
        )
        del policy

    print(f"saved comparison={args.output}", flush=True)


if __name__ == "__main__":
    main()
