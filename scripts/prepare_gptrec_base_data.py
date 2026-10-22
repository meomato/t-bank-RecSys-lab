"""Build the common KION layer and the KION-GPTRec-base atomic dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gptrec.data.kion_gptrec_base import DatasetConfig, prepare_kion_gptrec_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gptrec_base_data.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig.from_yaml(args.config)
    manifest = prepare_kion_gptrec_base(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[done] dataset={config.output_dir.resolve()}")


if __name__ == "__main__":
    main()

