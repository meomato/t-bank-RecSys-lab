"""Train and evaluate the calibrated KION reward-model ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gptrec.training.reward_trainer import RewardTrainingConfig, run_reward_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_reward_model.yaml"))
    args = parser.parse_args()
    result = run_reward_training(RewardTrainingConfig.from_yaml(args.config))
    print(json.dumps(result["aggregate_test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
