import pandas as pd
import torch

from gptrec.data.sequence_dataset import (
    GPTRecEvalDataset,
    GPTRecEventStore,
    GPTRecTrainDataset,
)


def make_store() -> GPTRecEventStore:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3],
            "item_token_id": [10, 20, 30, 40, 50, 60, 70, 71, 72, 73, 74, 75],
            "event_date": [
                "2021-08-01",
                "2021-08-01",
                "2021-08-02",
                "2021-08-10",
                "2021-08-01",
                "2021-08-10",
                "2021-08-01",
                "2021-08-02",
                "2021-08-03",
                "2021-08-04",
                "2021-08-05",
                "2021-08-06",
            ],
            "split": [
                "train",
                "train",
                "train",
                "validation",
                "train",
                "validation",
                "train",
                "train",
                "train",
                "train",
                "train",
                "train",
            ],
        }
    )
    return GPTRecEventStore.from_frame(frame, seed=42)


def test_train_dataset_excludes_singletons_and_overlaps_chunks() -> None:
    dataset = GPTRecTrainDataset(make_store(), max_sequence_length=4)

    assert len(dataset) == 3
    chunks = []
    for sample in dataset:
        chunks.append(sample["input_ids"][sample["attention_mask"].bool()])

    long_user_chunks = [chunk for chunk in chunks if torch.any(chunk >= 70)]
    assert len(long_user_chunks) == 2
    assert long_user_chunks[0][-1].item() == long_user_chunks[1][0].item()
    assert all(not torch.any(chunk == 50) for chunk in chunks)


def test_day_aware_masks_same_day_transition() -> None:
    dataset = GPTRecTrainDataset(
        make_store(),
        max_sequence_length=4,
        mode="day_aware",
    )
    sample = next(
        item
        for item in dataset
        if torch.any(item["input_ids"] == 10) and torch.any(item["input_ids"] == 20)
    )
    active_labels = sample["labels"][sample["attention_mask"].bool()]

    assert active_labels.tolist().count(-100) == 2
    assert 30 in active_labels.tolist()


def test_eval_uses_previous_singleton_as_context() -> None:
    dataset = GPTRecEvalDataset(
        make_store(),
        split="validation",
        max_sequence_length=4,
    )

    user_two = next(item for item in dataset if item["user_id"].item() == 2)
    context = user_two["input_ids"][user_two["attention_mask"].bool()]

    assert context.tolist() == [50]
    assert user_two["target_id"].item() == 60

