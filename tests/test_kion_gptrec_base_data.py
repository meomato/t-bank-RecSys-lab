import json

import pandas as pd

from gptrec.data.kion_gptrec_base import DatasetConfig, prepare_kion_gptrec_base


def test_preparation_uses_train_only_vocabulary(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    pd.DataFrame(
        {
            "user_id": [1, 2, 1, 1, 3],
            "item_id": [10, 20, 20, 30, 10],
            "last_watch_dt": [
                "2021-08-08",
                "2021-08-08",
                "2021-08-09",
                "2021-08-10",
                "2021-08-16",
            ],
            "total_dur": [100, 200, 300, 400, 500],
            "watched_pct": [10, 20, 30, 40, 50],
        }
    ).to_csv(raw_dir / "interactions.csv", index=False)
    pd.DataFrame(
        {
            "user_id": [1, 2],
            "age": ["25-34", "35-44"],
            "income": ["20-40", "40-60"],
            "sex": ["M", "F"],
            "kids_flg": [0, 1],
        }
    ).to_csv(raw_dir / "users.csv", index=False)
    pd.DataFrame(
        {
            "item_id": [10, 20, 30],
            "content_type": ["film", "film", "series"],
            "title": ["a", "b", "c"],
            "title_orig": ["a", "b", "c"],
            "release_year": [2019, 2020, 2021],
            "genres": ["drama", "comedy", "drama"],
            "countries": ["RU", "RU", "US"],
            "for_kids": [0, 0, 1],
            "age_rating": [12, 16, 6],
            "studios": ["s1", "s2", "s3"],
            "directors": ["d1", "d2", "d3"],
            "actors": ["x", "y", "z"],
            "description": ["one", "two", "three"],
            "keywords": ["k1", "k2", "k3"],
        }
    ).to_csv(raw_dir / "items.csv", index=False)

    config = DatasetConfig(
        name="KION-GPTRec-base-test",
        raw_dir=raw_dir,
        common_dir=tmp_path / "common",
        output_dir=tmp_path / "gptrec_base",
        train_end=pd.Timestamp("2021-08-08"),
        validation_end=pd.Timestamp("2021-08-15"),
    )
    manifest = prepare_kion_gptrec_base(config)

    vocabulary = pd.read_parquet(config.output_dir / "item_vocabulary.parquet")
    events = pd.read_parquet(config.output_dir / "events.parquet")
    common = pd.read_parquet(config.common_dir / "interactions.parquet")
    saved_manifest = json.loads((config.output_dir / "manifest.json").read_text())

    assert vocabulary["item_id"].tolist() == [10, 20]
    assert vocabulary["item_token_id"].tolist() == [1, 2]
    assert 30 not in events["item_id"].tolist()
    assert "item_token_id" not in common.columns
    assert manifest["vocabulary"]["vocab_size_with_pad"] == 3
    assert manifest["splits"]["validation"]["unseen_item_events"] == 1
    assert saved_manifest == manifest

