"""Downloads the 5 CTR-23 datasets used in the paper's demo table, by name, from suite 353."""
import argparse
import os
import time

import openml

CTR23_SUITE_ID = 353
TARGET_COL = "target"
DATA_ROOT = "data"

DEMO_DATASETS = [
    "naval_propulsion_plant",
    "video_transcoding",
    "auction_verification",
    "socmob",
    "airfoil_self_noise",
]


def _safe_name(value):
    s = str(value)
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)


def download_demo(data_root=DATA_ROOT, max_retries=3, retry_delay=5, force=False, datasets=None):
    remaining = set(datasets or DEMO_DATASETS)
    suite = openml.study.get_suite(CTR23_SUITE_ID)

    ok, skipped, failed = [], [], []
    for did in suite.data:
        if not remaining:
            break
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                dataset = openml.datasets.get_dataset(did, download_data=False)
                name = _safe_name(dataset.name)
                if name not in remaining:
                    break

                out_dir = os.path.join(data_root, name)
                out_path = os.path.join(out_dir, "raw.csv")
                if os.path.isfile(out_path) and not force:
                    skipped.append(name)
                    remaining.discard(name)
                    break

                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=dataset.default_target_attribute
                )
                df = X.copy()
                df[TARGET_COL] = y

                os.makedirs(out_dir, exist_ok=True)
                df.to_csv(out_path, index=False)
                print(f"[ok] {name}: {len(df)} rows -> {out_path}")
                ok.append(name)
                remaining.discard(name)
                break
            except Exception as e:
                last_err = str(e)
                if attempt < max_retries:
                    time.sleep(retry_delay)
        else:
            if last_err is not None:
                failed.append((did, last_err))

    if remaining:
        failed.extend((None, f"name not found: {name}") for name in remaining)

    print(f"Done: {len(ok)} downloaded, {len(skipped)} cached, {len(failed)} failed/missing.")
    return ok, skipped, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    download_demo(
        data_root=args.data_root, max_retries=args.max_retries,
        retry_delay=args.retry_delay, force=args.force, datasets=args.datasets,
    )
