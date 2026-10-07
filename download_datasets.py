"""Download the public demo datasets from OpenML into data/<dataset>/raw.csv."""

from pathlib import Path

import openml

DATASETS = {
    "naval_propulsion_plant": 44969,
    "video_transcoding": 44974,
    "auction_verification": 44958,
    "socmob": 44987,
    "airfoil_self_noise": 44957
    }

DATA_DIR = Path("data")


def download(name: str, dataset_id: int) -> None:
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        include_ignore_attribute=False,
    )
    df = X.copy()
    df["target"] = y

    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw.csv", index=False)
    print(f"[ok] {name} (OpenML {dataset_id}): {len(df)} rows, {df.shape[1] - 1} features")


if __name__ == "__main__":
    for name, dataset_id in DATASETS.items():
        download(name, dataset_id)
