"""Runs the Global vs. Data-Driven (CART-leaf) comparison on the public
demo datasets downloaded by download_datasets.py, and writes
demo_results.csv.
"""
import argparse
import os
import shutil

import pandas as pd

import data
import cart
import train_global_model
import train_cart_model
import evaluate
from cart_auto_tuning import is_dataset_viable_for_cart_leaf, compute_min_samples_leaf
from download_datasets import DATASETS

TARGET_COL = "target"
OUTPUT_CSV = "demo_results.csv"


def _load_raw(name, data_root="data"):
    raw_path = os.path.join(data_root, name, "raw.csv")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"{raw_path} not found -- run download_datasets.py first.")
    return pd.read_csv(raw_path)


def _fresh_dir(path):
    """Delete old models so every run trains from scratch."""
    if os.path.isdir(path):
        shutil.rmtree(path)


def main(datasets=None, data_root="data", output_path=OUTPUT_CSV):
    names = datasets or list(DATASETS.keys())
    print(f"Running Global vs. Data-Driven comparison on {len(names)} dataset(s): {names}")

    all_results = []
    for name in names:
        print(f"\n=== {name} ===")
        _load_raw(name, data_root=data_root)

        data_dir = os.path.join(data_root, name)
        figures_dir = os.path.join("figures", name)
        models_dir = os.path.join("models", name)
        results_dir = os.path.join("results", name)

        raw_csv = os.path.join(data_dir, "raw.csv")
        train_path = os.path.join(data_dir, "train_df.csv")
        test_path = os.path.join(data_dir, "test_df.csv")
        train_df, test_df = data.split(
            csv_path=raw_csv, target_col=TARGET_COL, train_path=train_path, test_path=test_path,
        )

        min_samples_leaf = compute_min_samples_leaf(len(train_df))
        if not is_dataset_viable_for_cart_leaf(len(train_df), min_samples_leaf):
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
        _fresh_dir(global_save_path)
        _fresh_dir(leaf_save_root)

        train_global_model.train(train_csv=train_path, target_col=TARGET_COL, save_path=global_save_path)
        train_cart_model.train(train_csv=train_leaf_path, target_col=TARGET_COL, save_root=leaf_save_root)

        overall_results, _ = evaluate.evaluate(
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
                        help="run only these dataset names (default: all in download_datasets.py)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()
    main(datasets=args.datasets, data_root=args.data_root, output_path=args.output)
