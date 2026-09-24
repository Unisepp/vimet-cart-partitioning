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

import data
import cart
import train_global_model
import train_cart_model
import evaluate
from cart_auto_tuning import is_dataset_viable_for_cart_leaf, compute_min_samples_leaf

TARGET_COL = "target"
DEFAULT_CHECK_CSV = "results/heterogeneity_check.csv"
OUTPUT_CSV = "clustered_selection_results.csv"


def _safe_name(value):
    s = str(value)
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)


def _global_model_exists(save_path):
    """True if save_path already holds a saved AutoGluon predictor."""
    return os.path.isfile(os.path.join(save_path, "predictor.pkl"))


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

        # Split -> CART -> train Global + per-leaf AutoGluon models -> evaluate,
        # all under this dataset's shared data/figures/models/results directory.
        safe_name = _safe_name(name)
        data_dir = os.path.join("data", safe_name)
        figures_dir = os.path.join("figures", safe_name)
        models_dir = os.path.join("models", safe_name)
        results_dir = os.path.join("results", safe_name)
        os.makedirs(data_dir, exist_ok=True)

        raw_csv = os.path.join(data_dir, "raw.csv")
        df.to_csv(raw_csv, index=False)
        train_path = os.path.join(data_dir, "train_df.csv")
        test_path = os.path.join(data_dir, "test_df.csv")
        train_df, test_df = data.split(
            csv_path=raw_csv, target_col=TARGET_COL, train_path=train_path, test_path=test_path,
        )

        effective_min_samples_leaf = compute_min_samples_leaf(len(train_df))
        if not is_dataset_viable_for_cart_leaf(len(train_df), effective_min_samples_leaf):
            print(f"[skip] {name}: only {len(train_df)} training rows.")
            continue

        cart_result = cart.run(train_df, test_df, target_col=TARGET_COL, figures_dir=figures_dir)
        if cart_result is None:
            print(f"[skip] {name}: see log above")
            continue

        train_leaf_path = os.path.join(data_dir, "train_df_with_leaf.csv")
        test_leaf_path = os.path.join(data_dir, "test_df_with_leaf.csv")
        cart_result["train_df"].to_csv(train_leaf_path, index=False)
        cart_result["test_df"].to_csv(test_leaf_path, index=False)

        global_save_path = os.path.join(models_dir, "global")
        leaf_save_root = os.path.join(models_dir, "per_leaf")

        if _global_model_exists(global_save_path):
            print(f"[reuse] Global model already exists at {global_save_path}, skipping retrain.")
        else:
            train_global_model.train(train_csv=train_path, target_col=TARGET_COL, save_path=global_save_path)
        train_cart_model.train(train_csv=train_leaf_path, target_col=TARGET_COL, save_root=leaf_save_root)

        overall_results, leaf_comparison_df = evaluate.evaluate(
            test_csv=test_leaf_path, target_col=TARGET_COL,
            global_save_path=global_save_path, leaf_save_root=leaf_save_root, results_dir=results_dir,
        )

        result = {
            "dataset": name,
            "best_leaf_count": cart_result["best_leaf_count"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "rmse_global": overall_results["Global"],
            "rmse_leaf": overall_results["Leaf Model (CART)"],
        }
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
