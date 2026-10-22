"""Instantiate GPTRec-base from the project YAML configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gptrec.models.gptrec_base import GPTRecBase, GPTRecBaseConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gptrec_base.yaml"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/processed/gptrec_base/manifest.json"),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for the random initial state. Normally omit this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.dataset_manifest.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    vocab_size = int(manifest["vocabulary"]["vocab_size_with_pad"])

    config = GPTRecBaseConfig.from_yaml(args.config, vocab_size=vocab_size)
    model = GPTRecBase(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"model={config.name}")
    print(f"parameters={parameter_count:,}")
    print(f"vocab_size={config.vocab_size:,}")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_dir / "model.pt")
        with (args.save_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(config.to_dict(), stream, ensure_ascii=False, indent=2)
        print(f"saved={args.save_dir.resolve()}")


if __name__ == "__main__":
    main()

