"""PyTorch datasets and data loaders for atomic-item GPTRec training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SequenceMode = Literal["standard", "day_aware"]

SPLIT_TO_CODE = {"train": 0, "validation": 1, "test": 2}
REQUIRED_EVENT_COLUMNS = ("user_id", "item_token_id", "event_date", "split")


@dataclass(frozen=True, slots=True)
class GPTRecEventStore:
    """Compact, chronologically ordered arrays shared by all dataset splits."""

    user_ids: np.ndarray
    item_token_ids: np.ndarray
    event_dates: np.ndarray
    split_codes: np.ndarray
    user_starts: np.ndarray
    user_ends: np.ndarray

    @classmethod
    def from_parquet(cls, path: str | Path, *, seed: int = 42) -> GPTRecEventStore:
        frame = pd.read_parquet(path, columns=list(REQUIRED_EVENT_COLUMNS))
        return cls.from_frame(frame, seed=seed)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, seed: int = 42) -> GPTRecEventStore:
        missing = sorted(set(REQUIRED_EVENT_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"events table is missing columns: {missing}")
        if frame.empty:
            raise ValueError("events table is empty")

        ordered = frame.loc[:, REQUIRED_EVENT_COLUMNS].copy()
        ordered["event_date"] = pd.to_datetime(ordered["event_date"], errors="raise")
        split_codes = ordered["split"].map(SPLIT_TO_CODE)
        if split_codes.isna().any():
            invalid = sorted(ordered.loc[split_codes.isna(), "split"].astype(str).unique())
            raise ValueError(f"unknown split values: {invalid}")

        tie_hash = pd.util.hash_pandas_object(
            ordered[["user_id", "event_date", "item_token_id"]],
            index=False,
        ).to_numpy(dtype=np.uint64)
        ordered["_tie_key"] = tie_hash ^ np.uint64(seed)
        ordered = ordered.sort_values(
            ["user_id", "event_date", "_tie_key"],
            kind="mergesort",
            ignore_index=True,
        )

        user_ids = ordered["user_id"].to_numpy(dtype=np.int64)
        changes = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
        user_starts = np.concatenate((np.array([0], dtype=np.int64), changes))
        user_ends = np.concatenate((changes, np.array([len(ordered)], dtype=np.int64)))

        split_codes = ordered["split"].map(SPLIT_TO_CODE).to_numpy(dtype=np.int8)
        return cls(
            user_ids=user_ids,
            item_token_ids=ordered["item_token_id"].to_numpy(dtype=np.int64),
            event_dates=ordered["event_date"].to_numpy(dtype="datetime64[D]"),
            split_codes=split_codes,
            user_starts=user_starts,
            user_ends=user_ends,
        )


class GPTRecTrainDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal-LM training chunks containing train events only.

    Long histories are split into chunks with one-token overlap so that the
    transition at each chunk boundary is retained exactly once.
    """

    def __init__(
        self,
        store: GPTRecEventStore,
        *,
        max_sequence_length: int = 32,
        mode: SequenceMode = "standard",
        pad_token_id: int = 0,
    ) -> None:
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        if mode not in ("standard", "day_aware"):
            raise ValueError(f"unknown sequence mode: {mode}")

        self.store = store
        self.max_sequence_length = max_sequence_length
        self.mode = mode
        self.pad_token_id = pad_token_id

        bounds: list[tuple[int, int]] = []
        for user_start, user_end in zip(store.user_starts, store.user_ends, strict=True):
            user_splits = store.split_codes[user_start:user_end]
            train_count = int(np.count_nonzero(user_splits == SPLIT_TO_CODE["train"]))
            train_end = int(user_start + train_count)
            chunk_start = int(user_start)

            while train_end - chunk_start >= 2:
                chunk_end = min(chunk_start + max_sequence_length, train_end)
                if mode == "standard" or np.any(
                    store.event_dates[chunk_start + 1 : chunk_end]
                    != store.event_dates[chunk_start : chunk_end - 1]
                ):
                    bounds.append((chunk_start, chunk_end))
                if chunk_end == train_end:
                    break
                chunk_start = chunk_end - 1

        self.sample_bounds = np.asarray(bounds, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.sample_bounds)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, end = self.sample_bounds[index]
        tokens = self.store.item_token_ids[start:end]
        dates = self.store.event_dates[start:end]
        length = len(tokens)

        input_ids = np.full(self.max_sequence_length, self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros(self.max_sequence_length, dtype=np.int64)
        labels = np.full(self.max_sequence_length, -100, dtype=np.int64)

        input_ids[:length] = tokens
        attention_mask[:length] = 1
        labels[:length] = tokens
        labels[0] = -100
        if self.mode == "day_aware":
            same_day_positions = np.flatnonzero(dates[1:] == dates[:-1]) + 1
            labels[same_day_positions] = -100

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "labels": torch.from_numpy(labels),
        }


class GPTRecEvalDataset(Dataset[dict[str, torch.Tensor]]):
    """One last eligible target per user for validation or test."""

    def __init__(
        self,
        store: GPTRecEventStore,
        *,
        split: Literal["validation", "test"],
        max_sequence_length: int = 32,
        mode: SequenceMode = "standard",
        pad_token_id: int = 0,
    ) -> None:
        if split not in ("validation", "test"):
            raise ValueError("evaluation split must be validation or test")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if mode not in ("standard", "day_aware"):
            raise ValueError(f"unknown sequence mode: {mode}")

        self.store = store
        self.max_sequence_length = max_sequence_length
        self.mode = mode
        self.pad_token_id = pad_token_id
        split_code = SPLIT_TO_CODE[split]

        samples: list[tuple[int, int]] = []
        for user_start, user_end in zip(store.user_starts, store.user_ends, strict=True):
            candidate_offsets = np.flatnonzero(
                store.split_codes[user_start:user_end] == split_code
            )
            if len(candidate_offsets) == 0:
                continue

            candidate_indices = candidate_offsets + user_start
            if mode == "standard":
                target_index = int(candidate_indices[-1])
            else:
                latest_date = store.event_dates[candidate_indices[-1]]
                latest_day_candidates = candidate_indices[
                    store.event_dates[candidate_indices] == latest_date
                ]
                target_index = int(latest_day_candidates[0])

            if target_index <= user_start:
                continue
            if mode == "day_aware" and (
                store.event_dates[target_index - 1] == store.event_dates[target_index]
            ):
                continue

            context_start = max(int(user_start), target_index - max_sequence_length)
            samples.append((context_start, target_index))

        self.samples = np.asarray(samples, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        context_start, target_index = self.samples[index]
        context = self.store.item_token_ids[context_start:target_index]
        length = len(context)

        input_ids = np.full(self.max_sequence_length, self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros(self.max_sequence_length, dtype=np.int64)
        input_ids[:length] = context
        attention_mask[:length] = 1

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "target_id": torch.tensor(
                self.store.item_token_ids[target_index],
                dtype=torch.long,
            ),
            "user_id": torch.tensor(self.store.user_ids[target_index], dtype=torch.long),
        }


@dataclass(frozen=True, slots=True)
class GPTRecDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def create_gptrec_dataloaders(
    events_path: str | Path,
    *,
    max_sequence_length: int = 32,
    batch_size: int = 32,
    mode: SequenceMode = "standard",
    seed: int = 42,
    num_workers: int = 0,
) -> GPTRecDataLoaders:
    """Create reproducible train, validation and test loaders."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    store = GPTRecEventStore.from_parquet(events_path, seed=seed)
    train_dataset = GPTRecTrainDataset(
        store,
        max_sequence_length=max_sequence_length,
        mode=mode,
    )
    validation_dataset = GPTRecEvalDataset(
        store,
        split="validation",
        max_sequence_length=max_sequence_length,
        mode=mode,
    )
    test_dataset = GPTRecEvalDataset(
        store,
        split="test",
        max_sequence_length=max_sequence_length,
        mode=mode,
    )
    generator = torch.Generator().manual_seed(seed)

    return GPTRecDataLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        validation=DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )

