"""Train KL-regularized GPTRec PPO policies in slate and feedback modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gptrec.training.ppo_trainer import PPOTrainingConfig, run_ppo_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_ppo.yaml"))
    parser.add_argument("--mode", choices=("slate", "feedback"))
    parser.add_argument(
        "--max-updates",
        type=int,
        help="Optional smoke/debug limit; the configured value is used otherwise.",
    )
    args = parser.parse_args()
    config = PPOTrainingConfig.from_yaml(args.config)
    results = run_ppo_training(config, mode=args.mode, max_updates=args.max_updates)
    summary = {
        run_mode: {
            "best_update": result["best_update"],
            "best_validation_score": result["best_validation_score"],
            "best_satisfies_ndcg_constraint": result[
                "best_satisfies_ndcg_constraint"
            ],
            "base_validation": result["base_validation"],
            "test": result["test"],
        }
        for run_mode, result in results.items()
    }
    print("\n=== PPO FINAL SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
