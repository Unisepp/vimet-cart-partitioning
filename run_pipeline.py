"""Runs the Global vs. Data-Driven (CART-leaf) comparison on the public
demo datasets downloaded by download_datasets.py, and writes
demo_results.csv (or demo_results_search.csv with --search).
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
import hyperparameter_search
from cart_auto_tuning import is_dataset_viable_for_cart_leaf, compute_min_samples_leaf
from download_datasets import DATASETS

TARGET_COL = "target"
OUTPUT_CSV = "demo_results.csv"
SEARCH_OUTPUT_CSV = "demo_results_search.csv"


def _load_raw(name, data_root="data"):
    raw_path = os.path.join(data_root, name, "raw.csv")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"{raw_path} not found -- run download_datasets.py first.")
    return pd.read_csv(raw_path)


def _fresh_dir(path):
    """Delete old models so every run trains from scratch."""
    if os.path.isdir(path):
        shutil.rmtree(path)


def main(datasets=None, data_root="data", output_path=None, search=False,
         search_time_limit=hyperparameter_search.SEARCH_TIME_LIMIT):
    names = datasets or list(DATASETS.keys())
    output_path = output_path or (SEARCH_OUTPUT_CSV if search else OUTPUT_CSV)
    tag = "_search" if search else ""
    mode = "with hyperparameter search" if search else "with default hyperparameters"
    print(f"Running Global vs. Data-Driven comparison {mode} on {len(names)} dataset(s): {names}")

    all_results = []
    for name in names:
        print(f"\n=== {name} ===")
        _load_raw(name, data_root=data_root)

        data_dir = os.path.join(data_root, name)
        figures_dir = os.path.join("figures", name + tag)
        models_dir = os.path.join("models", name + tag)
        results_dir = os.path.join("results", name + tag)

        raw_csv = os.path.join(data_dir, "raw.csv")
        train_path = os.path.join(data_dir, "train_df.csv")
        test_path = os.path.join(data_dir, "test_df.csv")
        train_df, test_df = data.split(
            csv_path=raw_csv, target_col=TARGET_COL, train_path=train_path, test_path=test_path,
        )

        hp = None
        if search:
            hp = hyperparameter_search.search(
                train_df, target_col=TARGET_COL,
                work_root=os.path.join(models_dir, "_search_tmp"),
                log_path=os.path.join(results_dir, "hp_search.csv"),
                time_limit=search_time_limit,
            )
            if hp is None:
                print(f"[skip] {name}: no viable hyperparameter setting.")
                continue
            min_samples_leaf = hp["min_samples_leaf"]
            leaf_grid = [hp["max_leaf_nodes"]]
        else:
            min_samples_leaf = compute_min_samples_leaf(len(train_df))
            leaf_grid = None
            if not is_dataset_viable_for_cart_leaf(len(train_df), min_samples_leaf):
                print(f"[skip] {name}: only {len(train_df)} training rows.")
                continue

        cart_result = cart.run(
            train_df, test_df, target_col=TARGET_COL, figures_dir=figures_dir,
            min_samples_leaf=min_samples_leaf, leaf_grid=leaf_grid,
        )
        if cart_result is None:
            print(f"[skip] {name}: see log above")
            continue

        train_leaf_path = os.path.join(data_dir, f"train_df_with_leaf{tag}.csv")
        test_leaf_path = os.path.join(data_dir, f"test_df_with_leaf{tag}.csv")
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
            "min_samples_leaf": cart_result["min_samples_leaf"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "rmse_global": overall_results["Global"],
            "rmse_leaf": overall_results["Leaf Model (CART)"],
        }
        if hp is not None:
            result["selection_rule"] = hp["rule"]
            result["search_val_rmse"] = hp["val_rmse"]
        result["samples"] = result["n_train"] + result["n_test"]
        result["improvement_pct"] = (
            (result["rmse_global"] - result["rmse_leaf"]) / result["rmse_global"] * 100
        )
        all_results.append(result)
        print(f"[ok] {name}: leaves={result['best_leaf_count']}, "
              f"min_samples_leaf={result['min_samples_leaf']}, "
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
    parser.add_argument("--output", default=None,
                        help="default: demo_results.csv, or demo_results_search.csv with --search")
    parser.add_argument("--search", action="store_true",
                        help="search min_samples_leaf, max_leaf_nodes and selection rule per dataset")
    parser.add_argument("--search-time-limit", type=int,
                        default=hyperparameter_search.SEARCH_TIME_LIMIT,
                        help="seconds per AutoGluon model during the search (default: 60)")
    args = parser.parse_args()
    main(datasets=args.datasets, data_root=args.data_root, output_path=args.output,
         search=args.search, search_time_limit=args.search_time_limit)
