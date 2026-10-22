"""Prepare the common KION layer and the atomic KION-GPTRec-base dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

INTERACTION_COLUMNS = (
    "user_id",
    "item_id",
    "last_watch_dt",
    "total_dur",
    "watched_pct",
)
USER_COLUMNS = ("user_id", "age", "income", "sex", "kids_flg")
ITEM_COLUMNS = (
    "item_id",
    "content_type",
    "title",
    "title_orig",
    "release_year",
    "genres",
    "countries",
    "for_kids",
    "age_rating",
    "studios",
    "directors",
    "actors",
    "description",
    "keywords",
)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Paths and fixed calendar boundaries for KION-GPTRec-base."""

    name: str
    raw_dir: Path
    common_dir: Path
    output_dir: Path
    train_end: pd.Timestamp
    validation_end: pd.Timestamp

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}

        dataset = document.get("dataset", {})
        split = document.get("split", {})
        config = cls(
            name=str(dataset["name"]),
            raw_dir=Path(dataset["raw_dir"]),
            common_dir=Path(dataset["common_dir"]),
            output_dir=Path(dataset["output_dir"]),
            train_end=pd.Timestamp(split["train_end"]),
            validation_end=pd.Timestamp(split["validation_end"]),
        )
        if config.train_end >= config.validation_end:
            raise ValueError("train_end must be earlier than validation_end")
        return config


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _require_unique_key(frame: pd.DataFrame, key: str, table: str) -> None:
    duplicate_count = int(frame[key].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{table}.{key} contains {duplicate_count:,} duplicate values")


def _split_stats(
    common: pd.DataFrame,
    events: pd.DataFrame,
    split_name: str,
) -> dict[str, int]:
    common_part = common.loc[common["split"] == split_name]
    event_part = events.loc[events["split"] == split_name]
    return {
        "source_events": int(len(common_part)),
        "gptrec_events": int(len(event_part)),
        "unseen_item_events": int(len(common_part) - len(event_part)),
        "users": int(event_part["user_id"].nunique()),
        "items": int(event_part["item_id"].nunique()),
        "eligible_next_item_targets": int((event_part["history_length"] > 0).sum()),
    }


def prepare_kion_gptrec_base(config: DatasetConfig) -> dict[str, Any]:
    """Create normalized common tables and the atomic GPTRec-base event table."""
    source_paths = {
        "interactions": config.raw_dir / "interactions.csv",
        "users": config.raw_dir / "users.csv",
        "items": config.raw_dir / "items.csv",
    }
    missing_files = [str(path) for path in source_paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing KION source files: {missing_files}")

    interactions = pd.read_csv(source_paths["interactions"], low_memory=False)
    users = pd.read_csv(source_paths["users"], low_memory=False)
    items = pd.read_csv(source_paths["items"], low_memory=False)

    _require_columns(interactions, INTERACTION_COLUMNS, "interactions")
    _require_columns(users, USER_COLUMNS, "users")
    _require_columns(items, ITEM_COLUMNS, "items")
    _require_unique_key(users, "user_id", "users")
    _require_unique_key(items, "item_id", "items")

    common = interactions.loc[:, INTERACTION_COLUMNS].copy()
    common = common.rename(columns={"last_watch_dt": "event_date"})
    common["event_date"] = pd.to_datetime(common["event_date"], errors="raise")
    common["split"] = np.select(
        [
            common["event_date"] <= config.train_end,
            common["event_date"] <= config.validation_end,
        ],
        ["train", "validation"],
        default="test",
    )
    common = common.sort_values(
        ["user_id", "event_date", "item_id"],
        kind="mergesort",
        ignore_index=True,
    )

    train_popularity = (
        common.loc[common["split"] == "train", "item_id"]
        .value_counts()
        .rename("train_interactions")
    )
    vocabulary = pd.DataFrame({"item_id": np.sort(train_popularity.index.to_numpy())})
    vocabulary["item_token_id"] = np.arange(1, len(vocabulary) + 1, dtype=np.int32)
    vocabulary["train_interactions"] = (
        vocabulary["item_id"].map(train_popularity).astype("int64")
    )

    token_by_item = vocabulary.set_index("item_id")["item_token_id"]
    common["item_token_id"] = common["item_id"].map(token_by_item).astype("Int32")
    event_columns = ["user_id", "item_id", "item_token_id", "event_date", "split"]
    events = common.loc[common["item_token_id"].notna(), event_columns].copy()
    events["item_token_id"] = events["item_token_id"].astype("int32")
    events["history_length"] = events.groupby("user_id", sort=False).cumcount().astype("int32")

    previous_user = common["user_id"].shift()
    previous_date = common["event_date"].shift()
    same_day_followup_events = int(
        ((common["user_id"] == previous_user) & (common["event_date"] == previous_date)).sum()
    )

    known_users = set(users["user_id"])
    known_items = set(items["item_id"])
    interaction_user_ids = set(common["user_id"])
    interaction_item_ids = set(common["item_id"])

    config.common_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    common.drop(columns="item_token_id").to_parquet(
        config.common_dir / "interactions.parquet",
        index=False,
        compression="zstd",
    )
    users.to_parquet(config.common_dir / "users.parquet", index=False, compression="zstd")
    items.to_parquet(config.common_dir / "items.parquet", index=False, compression="zstd")
    events.to_parquet(config.output_dir / "events.parquet", index=False, compression="zstd")
    vocabulary.to_parquet(
        config.output_dir / "item_vocabulary.parquet",
        index=False,
        compression="zstd",
    )

    manifest: dict[str, Any] = {
        "dataset_name": config.name,
        "description": "Atomic item-token events for GPTRec-base next-item prediction.",
        "source": "KION",
        "date_range": {
            "first": str(common["event_date"].min().date()),
            "last": str(common["event_date"].max().date()),
        },
        "split": {
            "train": f"event_date <= {config.train_end.date()}",
            "validation": (
                f"{config.train_end.date()} < event_date <= {config.validation_end.date()}"
            ),
            "test": f"event_date > {config.validation_end.date()}",
        },
        "vocabulary": {
            "source": "train only",
            "pad_token_id": 0,
            "item_count": int(len(vocabulary)),
            "vocab_size_with_pad": int(len(vocabulary) + 1),
            "unknown_item_policy": "Exclude from GPTRec events; retain in common interactions.",
        },
        "ordering": {
            "columns": ["user_id", "event_date", "item_id"],
            "same_day_tie_breaker": "item_id ascending because KION has no time-of-day field",
            "same_day_followup_events": same_day_followup_events,
        },
        "metadata_coverage": {
            "interaction_users_without_metadata": int(len(interaction_user_ids - known_users)),
            "interaction_items_without_metadata": int(len(interaction_item_ids - known_items)),
        },
        "splits": {
            name: _split_stats(common, events, name)
            for name in ("train", "validation", "test")
        },
        "files": {
            "common_interactions": str(config.common_dir / "interactions.parquet"),
            "common_users": str(config.common_dir / "users.parquet"),
            "common_items": str(config.common_dir / "items.parquet"),
            "gptrec_events": str(config.output_dir / "events.parquet"),
            "item_vocabulary": str(config.output_dir / "item_vocabulary.parquet"),
        },
    }
    with (config.output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    return manifest

