"""Datasets for KION reward modeling with strictly prior-day histories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

RewardSplit = Literal["train", "calibration", "test"]
REQUIRED_COLUMNS = (
    "user_id",
    "item_token_id",
    "event_date",
    "split",
    "completion_class",
    "duration_utility",
    "return_7d",
    "content_type_id",
)


@dataclass(frozen=True, slots=True)
class RewardEventStore:
    user_ids: np.ndarray
    item_token_ids: np.ndarray
    event_dates: np.ndarray
    split_names: np.ndarray
    completion_classes: np.ndarray
    duration_utilities: np.ndarray
    return_labels: np.ndarray
    content_type_ids: np.ndarray
    user_starts: np.ndarray
    user_ends: np.ndarray
    item_numeric_features: np.ndarray

    @classmethod
    def from_parquet(
        cls,
        events_path: str | Path,
        item_features_path: str | Path,
    ) -> RewardEventStore:
        events = pd.read_parquet(events_path, columns=list(REQUIRED_COLUMNS))
        item_features = pd.read_parquet(item_features_path)
        return cls.from_frames(events, item_features)

    @classmethod
    def from_frames(
        cls,
        events: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> RewardEventStore:
        missing = sorted(set(REQUIRED_COLUMNS).difference(events.columns))
        if missing:
            raise ValueError(f"reward events are missing columns: {missing}")
        if events.empty:
            raise ValueError("reward events are empty")
        ordered = events.loc[:, REQUIRED_COLUMNS].copy()
        ordered["event_date"] = pd.to_datetime(ordered["event_date"], errors="raise")
        ordered = ordered.sort_values(
            ["user_id", "event_date", "item_token_id"], kind="mergesort", ignore_index=True
        )
        user_ids = ordered["user_id"].to_numpy(dtype=np.int64)
        changes = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
        starts = np.concatenate(([0], changes)).astype(np.int64)
        ends = np.concatenate((changes, [len(ordered)])).astype(np.int64)

        max_token = int(
            max(ordered["item_token_id"].max(), item_features["item_token_id"].max())
        )
        numeric_columns = ["release_year_scaled", "log_train_popularity"] + sorted(
            column for column in item_features if column.startswith("genre_")
        )
        numeric = np.zeros((max_token + 1, len(numeric_columns)), dtype=np.float32)
        feature_tokens = item_features["item_token_id"].to_numpy(dtype=np.int64)
        numeric[feature_tokens] = item_features[numeric_columns].to_numpy(dtype=np.float32)
        completion = ordered["completion_class"].fillna(-1).to_numpy(dtype=np.int64)
        return cls(
            user_ids=user_ids,
            item_token_ids=ordered["item_token_id"].to_numpy(dtype=np.int64),
            event_dates=ordered["event_date"].to_numpy(dtype="datetime64[D]"),
            split_names=ordered["split"].astype(str).to_numpy(),
            completion_classes=completion,
            duration_utilities=ordered["duration_utility"].to_numpy(dtype=np.float32),
            return_labels=ordered["return_7d"].to_numpy(dtype=np.float32),
            content_type_ids=ordered["content_type_id"].to_numpy(dtype=np.int64),
            user_starts=starts,
            user_ends=ends,
            item_numeric_features=numeric,
        )


class KIONRewardDataset(Dataset[dict[str, torch.Tensor]]):
    """Observed actions with histories ending before the action's calendar day."""

    def __init__(
        self,
        store: RewardEventStore,
        *,
        split: RewardSplit,
        max_sequence_length: int = 32,
        pad_token_id: int = 0,
        max_examples: int | None = None,
    ) -> None:
        if split not in ("train", "calibration", "test"):
            raise ValueError(f"unsupported reward split: {split}")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.store = store
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

        samples: list[tuple[int, int, int]] = []
        for user_start, user_end in zip(store.user_starts, store.user_ends, strict=True):
            dates = store.event_dates[user_start:user_end]
            candidate_offsets = np.flatnonzero(
                store.split_names[user_start:user_end] == split
            )
            for offset in candidate_offsets:
                target_index = int(user_start + offset)
                prior_end = int(user_start + np.searchsorted(dates, dates[offset], side="left"))
                if prior_end <= user_start:
                    continue
                context_start = max(int(user_start), prior_end - max_sequence_length)
                samples.append((context_start, prior_end, target_index))
                if max_examples is not None and len(samples) >= max_examples:
                    break
            if max_examples is not None and len(samples) >= max_examples:
                break
        self.samples = np.asarray(samples, dtype=np.int64).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        context_start, context_end, target_index = self.samples[index]
        context = self.store.item_token_ids[context_start:context_end]
        length = len(context)
        input_ids = np.full(self.max_sequence_length, self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros(self.max_sequence_length, dtype=np.int64)
        input_ids[:length] = context
        attention_mask[:length] = 1
        action_id = int(self.store.item_token_ids[target_index])
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "action_id": torch.tensor(action_id, dtype=torch.long),
            "content_type_id": torch.tensor(
                self.store.content_type_ids[target_index], dtype=torch.long
            ),
            "item_numeric_features": torch.from_numpy(
                self.store.item_numeric_features[action_id]
            ),
            "completion_class": torch.tensor(
                self.store.completion_classes[target_index], dtype=torch.long
            ),
            "duration_utility": torch.tensor(
                self.store.duration_utilities[target_index], dtype=torch.float32
            ),
            "return_7d": torch.tensor(
                self.store.return_labels[target_index], dtype=torch.float32
            ),
            "user_id": torch.tensor(self.store.user_ids[target_index], dtype=torch.long),
        }
