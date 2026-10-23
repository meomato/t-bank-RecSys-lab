"""Prepare temporally honest KION data for conditional reward modeling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

COMPLETION_BOUNDS = np.asarray([-np.inf, 0, 10, 50, 89, 99, np.inf])


@dataclass(frozen=True, slots=True)
class RewardDataConfig:
    interactions_path: Path
    items_path: Path
    vocabulary_path: Path
    output_dir: Path
    train_end: pd.Timestamp
    embargo_end: pd.Timestamp
    calibration_end: pd.Timestamp
    test_end: pd.Timestamp
    observation_end: pd.Timestamp
    return_horizon_days: int = 7

    @classmethod
    def from_yaml(cls, path: str | Path) -> RewardDataConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        paths = document["paths"]
        split = document["split"]
        config = cls(
            interactions_path=Path(paths["interactions"]),
            items_path=Path(paths["items"]),
            vocabulary_path=Path(paths["vocabulary"]),
            output_dir=Path(paths["output_dir"]),
            train_end=pd.Timestamp(split["train_end"]),
            embargo_end=pd.Timestamp(split["embargo_end"]),
            calibration_end=pd.Timestamp(split["calibration_end"]),
            test_end=pd.Timestamp(split["test_end"]),
            observation_end=pd.Timestamp(split["observation_end"]),
            return_horizon_days=int(document.get("labels", {}).get("return_horizon_days", 7)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        boundaries = (
            self.train_end,
            self.embargo_end,
            self.calibration_end,
            self.test_end,
            self.observation_end,
        )
        if list(boundaries) != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
            raise ValueError("reward-model split boundaries must be strictly increasing")
        if self.return_horizon_days < 1:
            raise ValueError("return_horizon_days must be positive")
        if self.test_end + pd.Timedelta(days=self.return_horizon_days) > self.observation_end:
            raise ValueError("observation window is too short for test return labels")


def _assign_split(dates: pd.Series, config: RewardDataConfig) -> np.ndarray:
    return np.select(
        [
            dates <= config.train_end,
            dates <= config.embargo_end,
            dates <= config.calibration_end,
            dates <= config.test_end,
        ],
        ["train", "embargo", "calibration", "test"],
        default="future_only",
    )


def _completion_class(values: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int8")
    valid = values.notna()
    result.loc[valid] = np.digitize(values.loc[valid], COMPLETION_BOUNDS[1:-1], right=True)
    return result


def _duration_percentile(
    values: pd.Series,
    content_types: pd.Series,
    train_mask: pd.Series,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    log_values = np.log1p(values.to_numpy(dtype=np.float64))
    result = np.empty(len(values), dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, 101)
    knots: dict[str, list[float]] = {}
    global_reference = log_values[train_mask.to_numpy()]
    if len(global_reference) == 0:
        raise ValueError("reward data has no train durations")
    for content_type in sorted(content_types.astype(str).unique()):
        type_mask = content_types.astype(str).eq(content_type).to_numpy()
        reference = log_values[type_mask & train_mask.to_numpy()]
        if len(reference) == 0:
            reference = global_reference
        type_knots = np.quantile(reference, quantiles)
        knots[content_type] = type_knots.tolist()
        result[type_mask] = np.interp(log_values[type_mask], type_knots, quantiles)
    return result, knots


def _next_active_date(frame: pd.DataFrame) -> pd.Series:
    active_days = frame[["user_id", "event_date"]].drop_duplicates()
    active_days = active_days.sort_values(["user_id", "event_date"], kind="mergesort")
    active_days["next_active_date"] = active_days.groupby("user_id", sort=False)[
        "event_date"
    ].shift(-1)
    mapping = active_days.set_index(["user_id", "event_date"])["next_active_date"]
    keys = pd.MultiIndex.from_frame(frame[["user_id", "event_date"]])
    return pd.Series(mapping.reindex(keys).to_numpy(), index=frame.index)


def _item_features(
    items: pd.DataFrame,
    vocabulary: pd.DataFrame,
    *,
    genre_limit: int = 32,
) -> tuple[pd.DataFrame, list[str]]:
    features = vocabulary[["item_id", "item_token_id", "train_interactions"]].merge(
        items[["item_id", "content_type", "release_year", "genres"]],
        on="item_id",
        how="left",
        validate="one_to_one",
    )
    type_values = sorted(features["content_type"].fillna("unknown").astype(str).unique())
    type_to_id = {value: index for index, value in enumerate(type_values)}
    features["content_type_id"] = (
        features["content_type"].fillna("unknown").astype(str).map(type_to_id).astype("int8")
    )
    years = pd.to_numeric(features["release_year"], errors="coerce")
    median_year = float(years.median())
    features["release_year_scaled"] = ((years.fillna(median_year) - median_year) / 25).clip(
        -3, 3
    )
    features["log_train_popularity"] = np.log1p(features["train_interactions"])
    max_log_popularity = float(features["log_train_popularity"].max())
    features["log_train_popularity"] /= max_log_popularity
    genre_lists = features["genres"].fillna("").astype(str).str.split(",")
    genre_counts: dict[str, int] = {}
    for values in genre_lists:
        for raw_value in values:
            value = raw_value.strip().lower()
            if value:
                genre_counts[value] = genre_counts.get(value, 0) + 1
    genre_vocabulary = [
        value
        for value, _ in sorted(genre_counts.items(), key=lambda pair: (-pair[1], pair[0]))[
            :genre_limit
        ]
    ]
    genre_columns = []
    genre_sets = genre_lists.map(
        lambda values: {value.strip().lower() for value in values if value.strip()}
    )
    for index, genre in enumerate(genre_vocabulary):
        column = f"genre_{index:02d}"
        features[column] = genre_sets.map(
            lambda values, selected=genre: float(selected in values)
        ).astype("float32")
        genre_columns.append(column)
    columns = [
        "item_id",
        "item_token_id",
        "content_type",
        "content_type_id",
        "release_year_scaled",
        "log_train_popularity",
        *genre_columns,
    ]
    return features[columns], genre_vocabulary


def prepare_kion_reward_data(config: RewardDataConfig) -> dict[str, Any]:
    """Create outcome labels without treating missing user-item pairs as negatives."""
    interactions = pd.read_parquet(config.interactions_path)
    items = pd.read_parquet(config.items_path)
    vocabulary = pd.read_parquet(config.vocabulary_path)
    interactions["event_date"] = pd.to_datetime(interactions["event_date"], errors="raise")
    interactions = interactions.loc[interactions["event_date"] <= config.observation_end].copy()
    interactions = interactions.merge(
        vocabulary[["item_id", "item_token_id"]],
        on="item_id",
        how="inner",
        validate="many_to_one",
    )
    interactions = interactions.merge(
        items[["item_id", "content_type"]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )
    interactions["content_type"] = interactions["content_type"].fillna("unknown")
    interactions["split"] = _assign_split(interactions["event_date"], config)
    interactions = interactions.sort_values(
        ["user_id", "event_date", "item_token_id"], kind="mergesort", ignore_index=True
    )

    next_date = _next_active_date(interactions)
    gap_days = (next_date - interactions["event_date"]).dt.days
    interactions["return_7d"] = gap_days.le(config.return_horizon_days).astype("int8")
    interactions["completion_class"] = _completion_class(interactions["watched_pct"])
    train_mask = interactions["split"].eq("train")
    duration_utility, duration_knots = _duration_percentile(
        interactions["total_dur"], interactions["content_type"], train_mask
    )
    interactions["duration_utility"] = duration_utility

    item_features, genre_vocabulary = _item_features(items, vocabulary)
    content_type_id = item_features.set_index("item_token_id")["content_type_id"]
    interactions["content_type_id"] = (
        interactions["item_token_id"].map(content_type_id).astype("int8")
    )
    output_columns = [
        "user_id",
        "item_id",
        "item_token_id",
        "event_date",
        "split",
        "completion_class",
        "duration_utility",
        "return_7d",
        "content_type_id",
    ]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    interactions[output_columns].to_parquet(
        config.output_dir / "events.parquet", index=False, compression="zstd"
    )
    item_features.to_parquet(
        config.output_dir / "item_features.parquet", index=False, compression="zstd"
    )

    split_stats = {}
    for split in ("train", "embargo", "calibration", "test", "future_only"):
        part = interactions.loc[interactions["split"].eq(split)]
        split_stats[split] = {
            "events": int(len(part)),
            "users": int(part["user_id"].nunique()),
            "items": int(part["item_id"].nunique()),
        }
    manifest: dict[str, Any] = {
        "description": (
            "Conditional KION outcomes for reward modeling; "
            "no missing pair is a negative."
        ),
        "split": {
            "train": f"event_date <= {config.train_end.date()}",
            "embargo": f"{config.train_end.date()} < event_date <= {config.embargo_end.date()}",
            "calibration": (
                f"{config.embargo_end.date()} < event_date <= {config.calibration_end.date()}"
            ),
            "test": f"{config.calibration_end.date()} < event_date <= {config.test_end.date()}",
            "future_only": f"event_date > {config.test_end.date()}",
        },
        "return_horizon_days": config.return_horizon_days,
        "completion_classes": ["0", "1-10", "11-50", "51-89", "90-99", "100"],
        "duration_target": "train-CDF percentile of log1p(total_dur), by content_type",
        "duration_quantile_knots": duration_knots,
        "vocab_size_with_pad": int(vocabulary["item_token_id"].max() + 1),
        "item_feature_dim": int(
            len(
                [
                    column
                    for column in item_features.columns
                    if column in ("release_year_scaled", "log_train_popularity")
                    or column.startswith("genre_")
                ]
            )
        ),
        "genre_vocabulary": genre_vocabulary,
        "split_stats": split_stats,
        "limitations": [
            "No impression or logging-policy data; unobserved items are not negatives.",
            "Outcomes are conditional on observed consumption, not causal effects of exposure.",
            "Events within a calendar day are unordered; histories use prior days only.",
        ],
    }
    with (config.output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest
