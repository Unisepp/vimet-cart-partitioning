"""Runs the Global vs. Data-Driven (CART-leaf) model comparison on whichever datasets currently pass the heterogeneity check in
results/heterogeneity_check.csv (from heterogeneity_check.py):
real HDBSCAN cluster structure found, and noise_fraction_pct at or below MAX_NOISE_PCT threshold.
It leads to the paper's 6-dataset CTR23 demonstration set
(tab:demo-results): abalone, brazilian_houses, cps88wages, diamonds,
naval_propulsion_plant, video_transcoding.
"""
import argparse
import os

import pandas as pd

import run_pipeline as pipeline

DEFAULT_CHECK_CSV = "results/heterogeneity_check.csv"
OUTPUT_CSV = "clustered_selection_results.csv"


def get_dataset_list(check_csv=DEFAULT_CHECK_CSV):
    """heterogeneity_check.py only ever writes rows that already
    pass the heterogeneity check, so every dataset name in this CSV is
    eligible, no further filtering needed here."""
    df = pd.read_csv(check_csv)
    names = df["dataset"].tolist()
    print(f"{len(names)} dataset(s) in {check_csv}: {names}")
    return names


def _load_raw(name, data_root="data"):
    raw_path = os.path.join(data_root, name, "raw.csv")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"{raw_path} not found -- was this dataset ever downloaded?")
    return pd.read_csv(raw_path)


def main(datasets=None, check_csv=DEFAULT_CHECK_CSV, data_root="data", output_path=OUTPUT_CSV):
    names = datasets or get_dataset_list(check_csv=check_csv)
    if not names:
        print(f"No datasets to run -- {check_csv} is empty.")
        return None

    print(f"Running TRUE-default CART-leaf comparison on {len(names)} dataset(s): {names}")
    all_results = []
    for name in names:
        print(f"\n=== {name} (default fraction=0.05/floor=300/ceiling=800, se_multiplier=1.0, "
              f"reusing shared models/{name}/global) ===")
        df = _load_raw(name, data_root=data_root)
        result = pipeline.run_one_dataset(name, df, reuse_global=True)
        if result is None:
            print(f"[skip] {name}: see log above")
            continue
        result["samples"] = result["n_train"] + result["n_test"]
        result["improvement_pct"] = (
            (result["rmse_global"] - result["rmse_leaf"]) / result["rmse_global"] * 100
        )
        all_results.append(result)
        print(f"[ok] {name}: leaves={result['best_leaf_count']}, "
              f"improvement_pct={result['improvement_pct']:.2f}%")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(all_results)}/{len(names)} results to {output_path}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="override: run only these dataset names instead of "
                              "reading the list from --check-csv")
    parser.add_argument("--check-csv", default=DEFAULT_CHECK_CSV,
                         help="output of heterogeneity_check.py")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()
    main(datasets=args.datasets, check_csv=args.check_csv, data_root=args.data_root,
         output_path=args.output)
