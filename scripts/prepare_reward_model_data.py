"""Prepare temporally split KION outcomes for reward-model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gptrec.data.kion_reward_model import RewardDataConfig, prepare_kion_reward_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reward_model_data.yaml"))
    args = parser.parse_args()
    result = prepare_kion_reward_data(RewardDataConfig.from_yaml(args.config))
    summary = {
        "description": result["description"],
        "split": result["split"],
        "split_stats": result["split_stats"],
        "limitations": result["limitations"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
