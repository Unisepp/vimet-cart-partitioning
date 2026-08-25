"""Runs Global vs. Data-Driven on the 5 demo datasets and writes demo_results.csv."""
import os
import traceback

import pandas as pd

from cart_auto_tuning import is_dataset_viable_for_cart_leaf, compute_min_samples_leaf
import data_generic as data
import cart_generic as cart
import train_global_generic as train_global
import train_cart_leaf_generic as train_cart_leaf
import evaluate_generic as evaluate

TARGET_COL = "target"

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


def run_one_dataset(dataset_name, df, cart_min_samples_leaf=None, cart_leaf_grid=None):
    safe_name = _safe_name(dataset_name)
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

    effective_min_samples_leaf = (
        cart_min_samples_leaf if cart_min_samples_leaf is not None
        else compute_min_samples_leaf(len(train_df))
    )
    if not is_dataset_viable_for_cart_leaf(len(train_df), effective_min_samples_leaf):
        print(f"SKIP: only {len(train_df)} training rows.")
        return None

    cart_kwargs = {}
    if cart_min_samples_leaf is not None:
        cart_kwargs["min_samples_leaf"] = cart_min_samples_leaf
    if cart_leaf_grid is not None:
        cart_kwargs["leaf_grid"] = cart_leaf_grid
    cart_result = cart.run(train_df, test_df, target_col=TARGET_COL, figures_dir=figures_dir, **cart_kwargs)
    if cart_result is None:
        return None

    train_leaf_path = os.path.join(data_dir, "train_df_with_leaf.csv")
    test_leaf_path = os.path.join(data_dir, "test_df_with_leaf.csv")
    cart_result["train_df"].to_csv(train_leaf_path, index=False)
    cart_result["test_df"].to_csv(test_leaf_path, index=False)

    global_save_path = os.path.join(models_dir, "global")
    leaf_save_root = os.path.join(models_dir, "per_leaf")

    train_global.train(train_csv=train_path, target_col=TARGET_COL, save_path=global_save_path)
    train_cart_leaf.train(train_csv=train_leaf_path, target_col=TARGET_COL, save_root=leaf_save_root)

    overall_results, leaf_comparison_df = evaluate.evaluate(
        test_csv=test_leaf_path, target_col=TARGET_COL,
        global_save_path=global_save_path, leaf_save_root=leaf_save_root, results_dir=results_dir,
    )

    return {
        "dataset": dataset_name,
        "best_leaf_count": cart_result["best_leaf_count"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "rmse_global": overall_results["Global"],
        "rmse_leaf": overall_results["Leaf Model (CART)"],
    }


def main(datasets=None, data_root="data"):
    datasets = datasets or DEMO_DATASETS
    all_results, skipped = [], []

    for name in datasets:
        raw_path = os.path.join(data_root, name, "raw.csv")
        if not os.path.isfile(raw_path):
            print(f"[skip] {name}: not downloaded yet.")
            skipped.append({"dataset": name, "reason": "not downloaded"})
            continue

        df = pd.read_csv(raw_path)
        print(f"=== {name} ({len(df)} rows) ===")
        try:
            result = run_one_dataset(name, df)
            if result is None:
                skipped.append({"dataset": name, "reason": "see log above"})
            else:
                all_results.append(result)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()
            skipped.append({"dataset": name, "reason": str(e)})

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df["samples"] = results_df["n_train"] + results_df["n_test"]
        results_df["improvement_pct"] = (
            (results_df["rmse_global"] - results_df["rmse_leaf"]) / results_df["rmse_global"] * 100
        )

    print(f"Ran {len(all_results)}/{len(datasets)} datasets.")
    results_df.to_csv("demo_results.csv", index=False)
    return results_df


if __name__ == "__main__":
    import sys
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    main(datasets=only)
