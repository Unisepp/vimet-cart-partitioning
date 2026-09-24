"""Downloads all 33 CTR-23 datasets to be filtered by heterogeneity_check.py to the heterogeneous candidates.
It downloads each dataset from OpenML and writes each to data/<name>/raw.csv.
"""
import argparse
import os
import time

import openml

TARGET_COL = "target"
DATA_ROOT = "data"

CTR23_DATASETS = {
    "Moneyball": 41021,
    "QSAR_fish_toxicity": 44970,
    "abalone": 44956,
    "airfoil_self_noise": 44957,
    "auction_verification": 44958,
    "brazilian_houses": 44990,
    "california_housing": 44977,
    "cars": 44994,
    "concrete_compressive_strength": 44959,
    "cps88wages": 44984,
    "cpu_activity": 44978,
    "diamonds": 44979,
    "energy_efficiency": 44960,
    "fifa": 45012,
    "fps_benchmark": 44992,
    "geographical_origin_of_music": 44965,
    "grid_stability": 44973,
    "health_insurance": 44993,
    "kin8nm": 44980,
    "kings_county": 44989,
    "miami_housing": 44983,
    "naval_propulsion_plant": 44969,
    "physiochemical_protein": 44963,
    "pumadyn32nh": 44981,
    "red_wine": 44972,
    "sarcos": 44976,
    "socmob": 44987,
    "solar_flare": 44966,
    "space_ga": 45402,
    "superconductivity": 44964,
    "video_transcoding": 44974,
    "wave_energy": 44975,
    "white_wine": 44971,
}


def download_one(name, data_id, data_root=DATA_ROOT, max_retries=3, retry_delay=5, force=False):
    out_path = os.path.join(data_root, name, "raw.csv")
    if os.path.isfile(out_path) and not force:
        return "skipped (already exists)"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            dataset = openml.datasets.get_dataset(data_id, download_data=True)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute
            )
            df = X.copy()
            df[TARGET_COL] = y
            for is_cat, col in zip(categorical_indicator, attribute_names):
                if is_cat and col in df.columns:
                    df[col] = df[col].astype("category")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df.to_csv(out_path, index=False)
            return f"ok ({len(df)} rows)"
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(retry_delay)
    return f"FAILED: {last_err}"


def main(datasets=None, data_root=DATA_ROOT, force=False):
    names = datasets or list(CTR23_DATASETS.keys())
    results = {}
    for name in names:
        if name not in CTR23_DATASETS:
            results[name] = "unknown dataset name"
            continue
        print(f"=== {name} (id={CTR23_DATASETS[name]}) ===")
        status = download_one(name, CTR23_DATASETS[name], data_root=data_root, force=force)
        print(status)
        results[name] = status

    ok = sum(1 for s in results.values() if s.startswith("ok") or s.startswith("skipped"))
    print(f"\nDone: {ok}/{len(results)} datasets available in {data_root}/.")
    failed = {n: s for n, s in results.items() if s.startswith("FAILED") or s == "unknown dataset name"}
    if failed:
        print(f"{len(failed)} failed/unknown:")
        for n, s in failed.items():
            print(f"  {n}: {s}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="subset of dataset names to download (default: all 33)")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--force", action="store_true", help="redownload even if raw.csv exists")
    args = parser.parse_args()
    main(datasets=args.datasets, data_root=args.data_root, force=args.force)
