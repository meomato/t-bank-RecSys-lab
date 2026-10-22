"""Inspect GPTRec-base Dataset/DataLoader shapes without training the model."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gptrec.data.sequence_dataset import create_gptrec_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/processed/gptrec_base/events.parquet"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/gptrec_base.yaml"),
    )
    parser.add_argument("--mode", choices=("standard", "day_aware"), default="standard")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.model_config.open("r", encoding="utf-8") as stream:
        model_config = (yaml.safe_load(stream) or {})["model"]

    loaders = create_gptrec_dataloaders(
        args.events,
        max_sequence_length=int(model_config["max_sequence_length"]),
        batch_size=args.batch_size,
        mode=args.mode,
    )
    print(f"mode={args.mode}")
    print(f"train_samples={len(loaders.train.dataset):,}")
    print(f"validation_samples={len(loaders.validation.dataset):,}")
    print(f"test_samples={len(loaders.test.dataset):,}")

    train_batch = next(iter(loaders.train))
    validation_batch = next(iter(loaders.validation))
    print(f"train_input_shape={tuple(train_batch['input_ids'].shape)}")
    print(f"train_labels_shape={tuple(train_batch['labels'].shape)}")
    print(f"validation_input_shape={tuple(validation_batch['input_ids'].shape)}")
    print(f"validation_target_shape={tuple(validation_batch['target_id'].shape)}")


if __name__ == "__main__":
    main()

