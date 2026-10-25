"""Plot the unified full-test PPO reward/relevance and collapse trade-offs."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


LABELS = {
    "gptrec_base": "GPTRec-base",
    "ppo_slate_beta_005_best": "Slate β=.05",
    "ppo_feedback_beta_005_best": "Feedback β=.05",
    "ppo_slate_beta_05_update_050": "Slate β=.5 · 50",
    "ppo_slate_beta_05_best": "Slate β=.5 · 200",
    "ppo_feedback_beta_05_update_050": "Feedback β=.5 · 50",
    "ppo_feedback_beta_05_best": "Feedback β=.5 · 250",
}

COLORS = {
    "gptrec_base": "#334155",
    "ppo_slate_beta_005_best": "#dc2626",
    "ppo_feedback_beta_005_best": "#dc2626",
    "ppo_slate_beta_05_update_050": "#16a34a",
    "ppo_feedback_beta_05_update_050": "#16a34a",
    "ppo_slate_beta_05_best": "#f59e0b",
    "ppo_feedback_beta_05_best": "#f59e0b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/generated/ppo_comparison/full_test.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/ppo-reward-relevance-tradeoff.png"),
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as stream:
        policies = json.load(stream)["policies"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    markers = {"gptrec_base": "*", "slate": "o", "feedback": "^"}
    for name, label in LABELS.items():
        row = policies[name]
        metrics = row["metrics"]
        marker = markers["gptrec_base" if name == "gptrec_base" else row["mode"]]
        size = 170 if name == "gptrec_base" else 85
        for axis, x_name, y_name in (
            (axes[0], "ndcg@5", "raw_reward_per_step"),
            (axes[1], "entropy_per_step", "unique_slate_rate"),
        ):
            axis.scatter(
                metrics[x_name],
                metrics[y_name],
                s=size,
                marker=marker,
                color=COLORS[name],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            offset = (6, 5) if "feedback" not in name else (6, -13)
            axis.annotate(
                label,
                (metrics[x_name], metrics[y_name]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                color="#0f172a",
            )

    axes[0].set(
        title="Компромисс relevance–reward",
        xlabel="NDCG@5  → лучше",
        ylabel="Raw reward  → лучше",
    )
    axes[1].set(
        title="Диагностика policy collapse",
        xlabel="Entropy  → разнообразнее распределение",
        ylabel="Доля уникальных Top-5 списков  → лучше",
    )
    for axis in axes:
        axis.margins(x=0.06, y=0.12)
        axis.grid(alpha=0.2, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "PPO-рекомендатели на едином полном test · 183 534 истории",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Круг — slate, треугольник — feedback; зелёный — ранний β=.5, "
        "оранжевый — выбранный β=.5, красный — длинный β=.05.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved plot={args.output}")


if __name__ == "__main__":
    main()
