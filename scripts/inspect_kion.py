"""Inspect raw KION tables and create a local graphical EDA report."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

RAW_DIR = Path("downloads/kion/raw")
REPORT_DIR = Path("reports/generated/kion_eda")

EXPECTED_FILES = {
    "interactions": RAW_DIR / "data_original" / "interactions.csv",
    "users": RAW_DIR / "data_original" / "users.csv",
    "items": RAW_DIR / "data_original" / "items.csv",
}


def require_files() -> None:
    missing = [str(path) for path in EXPECTED_FILES.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "KION files were not found. Run scripts/download_kion.py first.\n"
            f"Missing files:\n{formatted}"
        )


def load_tables() -> dict[str, pd.DataFrame]:
    print("[read] interactions.csv")
    interactions = pd.read_csv(EXPECTED_FILES["interactions"], low_memory=False)
    print("[read] users.csv")
    users = pd.read_csv(EXPECTED_FILES["users"], low_memory=False)
    print("[read] items.csv")
    items = pd.read_csv(EXPECTED_FILES["items"], low_memory=False)
    return {"interactions": interactions, "users": users, "items": items}


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def schema_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = len(frame)
    result = pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "missing": [int(frame[column].isna().sum()) for column in frame.columns],
            "missing_pct": [
                round(float(frame[column].isna().mean() * 100), 3)
                for column in frame.columns
            ],
            "unique": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        }
    )
    result.insert(1, "rows", rows)
    return result


def save_figure(fig: plt.Figure, filename: str) -> str:
    path = REPORT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_time(interactions: pd.DataFrame, date_column: str) -> tuple[str, pd.Series]:
    dates = pd.to_datetime(interactions[date_column], errors="coerce")
    daily = dates.dropna().dt.floor("D").value_counts().sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0)
    daily.index.name = "date"
    daily.name = "interactions"
    daily.to_csv(REPORT_DIR / "daily_interactions.csv")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(daily.index, daily.values, linewidth=1, alpha=0.45, label="daily")
    axes[0].plot(
        daily.index,
        daily.rolling(7, min_periods=1).mean(),
        linewidth=2,
        label="7-day average",
    )
    axes[0].set_title("KION interactions over calendar time")
    axes[0].set_ylabel("Interactions per day")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    cumulative = daily.cumsum() / daily.sum()
    axes[1].plot(cumulative.index, cumulative.values, linewidth=2)
    axes[1].axhline(0.8, color="tab:orange", linestyle="--", linewidth=1, label="80%")
    axes[1].axhline(0.9, color="tab:green", linestyle="--", linewidth=1, label="90%")
    axes[1].set_title("Cumulative share of interactions")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative share")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    return save_figure(fig, "interactions_over_time.png"), daily


def plot_activity(
    interactions: pd.DataFrame,
    user_column: str,
    item_column: str,
) -> tuple[str, pd.Series, pd.Series]:
    user_activity = interactions[user_column].value_counts()
    item_popularity = interactions[item_column].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(np.log10(user_activity.values), bins=50)
    axes[0].set_title("User history lengths")
    axes[0].set_xlabel("log10(interactions per user)")
    axes[0].set_ylabel("Users")
    axes[0].grid(alpha=0.2)

    axes[1].hist(np.log10(item_popularity.values), bins=50)
    axes[1].set_title("Item popularity")
    axes[1].set_xlabel("log10(interactions per item)")
    axes[1].set_ylabel("Items")
    axes[1].grid(alpha=0.2)
    return (
        save_figure(fig, "activity_distributions.png"),
        user_activity,
        item_popularity,
    )


def plot_missing_values(schemas: dict[str, pd.DataFrame]) -> str:
    fig, axes = plt.subplots(len(schemas), 1, figsize=(12, 3.5 * len(schemas)))
    axes = np.atleast_1d(axes)

    for axis, (name, schema) in zip(axes, schemas.items(), strict=True):
        view = schema.sort_values("missing_pct", ascending=False).head(20)
        axis.barh(view["column"], view["missing_pct"])
        axis.invert_yaxis()
        axis.set_title(f"Missing values: {name}")
        axis.set_xlabel("Missing values, %")
        axis.grid(axis="x", alpha=0.2)

    return save_figure(fig, "missing_values.png")


def plot_feedback(interactions: pd.DataFrame) -> str | None:
    watched_column = find_column(interactions, ("watched_pct", "played_ratio_pct"))
    duration_column = find_column(interactions, ("total_dur", "duration", "watch_duration"))
    selected = [column for column in (watched_column, duration_column) if column is not None]
    if not selected:
        return None

    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 4.5))
    axes = np.atleast_1d(axes)
    for axis, column in zip(axes, selected, strict=True):
        values = pd.to_numeric(interactions[column], errors="coerce").dropna()
        lower, upper = values.quantile([0.005, 0.995])
        clipped = values.clip(lower=lower, upper=upper)
        axis.hist(clipped, bins=50)
        axis.set_title(f"Distribution of {column}")
        axis.set_xlabel(f"{column} (0.5–99.5 percentile range)")
        axis.set_ylabel("Interactions")
        axis.grid(alpha=0.2)

    return save_figure(fig, "feedback_distributions.png")


def plot_item_metadata(items: pd.DataFrame) -> str | None:
    content_column = find_column(items, ("content_type", "type"))
    genres_column = find_column(items, ("genres", "genre"))
    release_column = find_column(items, ("release_year", "year"))
    plots = sum(column is not None for column in (content_column, genres_column, release_column))
    if plots == 0:
        return None

    fig, axes = plt.subplots(1, plots, figsize=(6 * plots, 4.8))
    axes = iter(np.atleast_1d(axes))

    if content_column is not None:
        axis = next(axes)
        counts = items[content_column].fillna("<missing>").astype(str).value_counts().head(15)
        axis.barh(counts.index[::-1], counts.values[::-1])
        axis.set_title("Content types")
        axis.set_xlabel("Items")
        axis.grid(axis="x", alpha=0.2)

    if genres_column is not None:
        axis = next(axes)
        genres = (
            items[genres_column]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )
        counts = genres[genres.ne("")].value_counts().head(15)
        axis.barh(counts.index[::-1], counts.values[::-1])
        axis.set_title("Top genres")
        axis.set_xlabel("Items")
        axis.grid(axis="x", alpha=0.2)

    if release_column is not None:
        axis = next(axes)
        years = pd.to_numeric(items[release_column], errors="coerce").dropna()
        years = years[years.between(1900, 2100)]
        axis.hist(years, bins=40)
        axis.set_title("Release years")
        axis.set_xlabel("Year")
        axis.set_ylabel("Items")
        axis.grid(alpha=0.2)

    return save_figure(fig, "item_metadata.png")


def quantiles(series: pd.Series) -> str:
    values = series.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).rename("value")
    values.index = ["min", "25%", "median", "75%", "90%", "99%", "max"]
    return values.to_frame().to_html(classes="dataframe", border=0)


def write_report(
    tables: dict[str, pd.DataFrame],
    schemas: dict[str, pd.DataFrame],
    figures: list[str],
    date_column: str,
    user_column: str,
    item_column: str,
    daily: pd.Series,
    user_activity: pd.Series,
    item_popularity: pd.Series,
) -> None:
    interactions = tables["interactions"]
    parsed_dates = pd.to_datetime(interactions[date_column], errors="coerce")
    duplicate_pairs = int(interactions.duplicated([user_column, item_column]).sum())

    overview = pd.DataFrame(
        [
            ("Interactions", f"{len(interactions):,}"),
            ("Users in interactions", f"{interactions[user_column].nunique():,}"),
            ("Items in interactions", f"{interactions[item_column].nunique():,}"),
            ("Users metadata rows", f"{len(tables['users']):,}"),
            ("Items metadata rows", f"{len(tables['items']):,}"),
            ("First interaction date", str(parsed_dates.min().date())),
            ("Last interaction date", str(parsed_dates.max().date())),
            ("Calendar days", f"{len(daily):,}"),
            ("Invalid/missing dates", f"{parsed_dates.isna().sum():,}"),
            ("Repeated user-item pairs", f"{duplicate_pairs:,}"),
        ],
        columns=["Metric", "Value"],
    )

    sections = [
        "<h1>KION raw-data inspection</h1>",
        "<p>This report describes raw files only. It does not remove columns or create splits.</p>",
        "<h2>Dataset overview</h2>",
        overview.to_html(index=False, classes="dataframe", border=0, escape=True),
    ]
    for figure in figures:
        sections.append(f'<img src="{html.escape(figure)}" alt="{html.escape(figure)}">')

    sections.extend(
        [
            "<h2>User activity quantiles</h2>",
            quantiles(user_activity),
            "<h2>Item popularity quantiles</h2>",
            quantiles(item_popularity),
        ]
    )
    for name, schema in schemas.items():
        sections.extend(
            [
                f"<h2>{html.escape(name.title())} columns</h2>",
                schema.to_html(index=False, classes="dataframe", border=0, escape=True),
            ]
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KION raw-data inspection</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: 32px auto; padding: 0 20px; }}
    img {{ display: block; max-width: 100%; margin: 28px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 7px; text-align: left; }}
    th {{ position: sticky; top: 0; background: #f3f3f3; }}
  </style>
</head>
<body>
{''.join(sections)}
</body>
</html>
"""
    (REPORT_DIR / "report.html").write_text(document, encoding="utf-8")


def main() -> None:
    require_files()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    interactions = tables["interactions"]

    user_column = find_column(interactions, ("user_id", "uid"))
    item_column = find_column(interactions, ("item_id", "movie_id"))
    date_column = find_column(interactions, ("last_watch_dt", "timestamp", "date"))
    if user_column is None or item_column is None or date_column is None:
        raise ValueError(
            "Could not detect user, item or date columns in interactions.csv. "
            f"Found columns: {list(interactions.columns)}"
        )

    schemas = {name: schema_summary(frame) for name, frame in tables.items()}
    for name, schema in schemas.items():
        schema.to_csv(REPORT_DIR / f"{name}_schema.csv", index=False)

    figures: list[str] = []
    time_figure, daily = plot_time(interactions, date_column)
    figures.append(time_figure)
    activity_figure, user_activity, item_popularity = plot_activity(
        interactions,
        user_column,
        item_column,
    )
    figures.append(activity_figure)
    figures.append(plot_missing_values(schemas))

    for optional_figure in (plot_feedback(interactions), plot_item_metadata(tables["items"])):
        if optional_figure is not None:
            figures.append(optional_figure)

    write_report(
        tables,
        schemas,
        figures,
        date_column,
        user_column,
        item_column,
        daily,
        user_activity,
        item_popularity,
    )
    print(f"[done] open {(REPORT_DIR / 'report.html').resolve()}")


if __name__ == "__main__":
    main()

