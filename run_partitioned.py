"""Runs PartitionedRegressor on the public demo datasets (random splits) and
writes demo_results_partitioned.csv plus one search log per dataset.

No global model is trained here. If run_pipeline.py has already trained one
(models/<dataset>/global), it is loaded to report the improvement over it.
"""
import argparse
import os

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

import data
from download_datasets import DATASETS
from partitioned_regressor import PartitionedRegressor, rmse

TARGET_COL = "target"
OUTPUT_CSV = "demo_results_partitioned.csv"


def main(datasets=None, data_root="data", output_path=OUTPUT_CSV, search_time_limit=None):
    rows = []
    for name in datasets or list(DATASETS):
        print(f"\n=== {name} ===")
        data_dir = os.path.join(data_root, name)
        train_df, test_df = data.split(
            csv_path=os.path.join(data_dir, "raw.csv"), target_col=TARGET_COL,
            train_path=os.path.join(data_dir, "train_df.csv"),
            test_path=os.path.join(data_dir, "test_df.csv"),
        )

        model = PartitionedRegressor(
            label=TARGET_COL, path=os.path.join("models", f"{name}_partitioned"),
            split="random", search_time_limit=search_time_limit,
        ).fit(train_df)
        rmse_partitioned = model.evaluate(test_df)["rmse"]

        global_path = os.path.join("models", name, "global")
        if os.path.isdir(global_path):
            global_model = TabularPredictor.load(global_path)
            rmse_global = rmse(test_df[TARGET_COL], global_model.predict(test_df.drop(columns=[TARGET_COL])))
        else:
            print(f"[info] no global model at {global_path}; run run_pipeline.py to compare.")
            rmse_global = np.nan

        os.makedirs("results", exist_ok=True)
        model.search_results_.to_csv(os.path.join("results", f"{name}_partitioned_search.csv"), index=False)

        rows.append({
            "dataset": name,
            "chosen": model.chosen_["source"],
            "min_samples_leaf": model.chosen_["min_samples_leaf"],
            "leaves": model._final["tree"].get_n_leaves(),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "rmse_partitioned": rmse_partitioned,
            "rmse_global": rmse_global,
            "improvement_pct": (rmse_global - rmse_partitioned) / rmse_global * 100,
        })
        print(f"[ok] {name}: {model.summary()}, test RMSE {rmse_partitioned:.6g}")

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)
    print(f"\nWrote {len(rows)} result(s) to {output_path}")
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--search-time-limit", type=int, default=None,
                        help="seconds per AutoGluon model during the search (default: no limit)")
    args = parser.parse_args()
    main(datasets=args.datasets, data_root=args.data_root, output_path=args.output,
         search_time_limit=args.search_time_limit)
