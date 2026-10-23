"""Plot reward-model ensemble training curves from the saved result.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _mean_curve(members: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray([row["epoch"] for row in members[0]["history"]], dtype=float)
    values = np.asarray(
        [[row[metric] for row in member["history"]] for member in members],
        dtype=float,
    )
    return epochs, values.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("reports/generated/reward_model/result.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/reward-model-training-curves.png"),
    )
    args = parser.parse_args()

    with args.result.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    members = result["members"]
    if not members:
        raise ValueError("reward-model result contains no ensemble members")

    colors = plt.cm.Blues(np.linspace(0.40, 0.85, len(members)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    panels = (
        ("train_loss", "Train loss", "Loss"),
        ("calibration_objective", "Calibration objective", "NLL + MAE + log loss"),
    )
    for axis, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for color, member in zip(colors, members, strict=True):
            epochs = [row["epoch"] for row in member["history"]]
            values = [row[metric] for row in member["history"]]
            axis.plot(
                epochs,
                values,
                color=color,
                marker="o",
                linewidth=1.4,
                alpha=0.72,
                label=f"seed {member['seed']}",
            )
        mean_epochs, mean_values = _mean_curve(members, metric)
        axis.plot(
            mean_epochs,
            mean_values,
            color="#111827",
            marker="o",
            linewidth=2.8,
            label="ensemble mean",
            zorder=5,
        )
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel, xticks=mean_epochs)
        axis.grid(alpha=0.22)

    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Reward model ensemble training", fontsize=15, fontweight="semibold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
