import pandas as pd

from gptrec.data.kion_reward_model import RewardDataConfig, prepare_kion_reward_data
from gptrec.data.reward_dataset import KIONRewardDataset, RewardEventStore


def test_reward_data_uses_prior_days_and_mature_return_labels(tmp_path) -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2],
            "item_id": [10, 20, 30, 40, 10, 30],
            "event_date": [
                "2021-08-01",
                "2021-08-01",
                "2021-08-10",
                "2021-08-20",
                "2021-08-01",
                "2021-08-13",
            ],
            "total_dur": [10, 100, 1000, 10000, 20, 2000],
            "watched_pct": [0, 10, 50, 100, 5, 90],
            "split": ["unused"] * 6,
        }
    )
    items = pd.DataFrame(
        {
            "item_id": [10, 20, 30, 40],
            "content_type": ["film", "film", "series", "series"],
            "release_year": [2018, 2019, 2020, 2021],
            "genres": ["drama", "comedy", "drama", "comedy"],
        }
    )
    vocabulary = pd.DataFrame(
        {
            "item_id": [10, 20, 30, 40],
            "item_token_id": [1, 2, 3, 4],
            "train_interactions": [10, 8, 6, 4],
        }
    )
    interactions_path = tmp_path / "interactions.parquet"
    items_path = tmp_path / "items.parquet"
    vocabulary_path = tmp_path / "vocabulary.parquet"
    interactions.to_parquet(interactions_path)
    items.to_parquet(items_path)
    vocabulary.to_parquet(vocabulary_path)
    config = RewardDataConfig(
        interactions_path=interactions_path,
        items_path=items_path,
        vocabulary_path=vocabulary_path,
        output_dir=tmp_path / "output",
        train_end=pd.Timestamp("2021-08-01"),
        embargo_end=pd.Timestamp("2021-08-08"),
        calibration_end=pd.Timestamp("2021-08-11"),
        test_end=pd.Timestamp("2021-08-15"),
        observation_end=pd.Timestamp("2021-08-22"),
    )

    manifest = prepare_kion_reward_data(config)
    events = pd.read_parquet(config.output_dir / "events.parquet")
    store = RewardEventStore.from_parquet(
        config.output_dir / "events.parquet",
        config.output_dir / "item_features.parquet",
    )
    calibration = KIONRewardDataset(store, split="calibration")

    assert manifest["split_stats"]["calibration"]["events"] == 1
    assert len(calibration) == 1
    sample = calibration[0]
    history = sample["input_ids"][sample["attention_mask"].bool()]
    assert history.tolist() == [1, 2]
    assert sample["action_id"].item() == 3
    # The next active day is ten days later, so it is not a 7-day return.
    assert sample["return_7d"].item() == 0
    assert events.loc[events["item_id"].eq(20), "completion_class"].item() == 1


def test_same_day_actions_share_prior_day_history() -> None:
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "item_token_id": [1, 2, 3],
            "event_date": ["2021-08-01", "2021-08-10", "2021-08-10"],
            "split": ["train", "calibration", "calibration"],
            "completion_class": [1, 2, 3],
            "duration_utility": [0.1, 0.2, 0.3],
            "return_7d": [1, 0, 0],
            "content_type_id": [0, 0, 1],
        }
    )
    features = pd.DataFrame(
        {
            "item_token_id": [1, 2, 3],
            "release_year_scaled": [0.0, 0.1, 0.2],
            "log_train_popularity": [1.0, 0.8, 0.6],
        }
    )
    dataset = KIONRewardDataset(
        RewardEventStore.from_frames(events, features), split="calibration"
    )
    histories = [
        sample["input_ids"][sample["attention_mask"].bool()].tolist()
        for sample in dataset
    ]
    assert histories == [[1], [1]]
