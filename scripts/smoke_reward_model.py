"""Run a small end-to-end reward-model training smoke test."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from gptrec.training.reward_trainer import RewardTrainingConfig, train_reward_member


def main() -> None:
    config = RewardTrainingConfig.from_yaml(Path("configs/train_reward_model.yaml"))
    config = replace(
        config,
        checkpoint_dir=Path("artifacts/checkpoints/reward_model_smoke"),
        report_dir=Path("reports/generated/reward_model_smoke"),
    )
    result = train_reward_member(
        config,
        seed=config.seeds[0],
        max_examples=4096,
        max_steps_per_epoch=10,
    )
    print(json.dumps(result["test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
