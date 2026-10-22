"""Train GPTRec-base and evaluate the best checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from gptrec.training import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_gptrec_base.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    config = TrainingConfig.from_yaml(parse_args().config)
    run_training(config)


if __name__ == "__main__":
    main()

