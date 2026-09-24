"""Scores Global vs. Data-Driven on the same test set; reports overall and per-leaf RMSE."""
import os
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset
from sklearn.metrics import mean_squared_error

import train_global_model
import train_cart_model

TARGET = "target"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(test_csv="data/test_df_with_leaf.csv", target_col=TARGET,
             global_save_path=train_global_model.SAVE_PATH,
             leaf_save_root=train_cart_model.SAVE_ROOT,
             results_dir="results"):
    test_df = pd.read_csv(test_csv)

    global_model = train_global_model.load(save_path=global_save_path)
    leaf_models = train_cart_model.load(save_root=leaf_save_root, train_df=test_df)

    test_df["pred_global"] = global_model.predict(TabularDataset(test_df.drop(columns=["leaf_id"]))).values

    test_df["pred_leaf"] = np.nan
    for leaf_id, predictor in leaf_models.items():
        mask = test_df["leaf_id"] == leaf_id
        X_leaf = test_df.loc[mask].drop(columns=["leaf_id", "pred_global", "pred_leaf"])
        preds = predictor.predict(TabularDataset(X_leaf))
        test_df.loc[mask, "pred_leaf"] = preds.values

    overall_results = {
        "Global": rmse(test_df[target_col], test_df["pred_global"]),
        "Leaf Model (CART)": rmse(test_df[target_col], test_df["pred_leaf"]),
    }

    leaf_rows = []
    for leaf_id, group in test_df.groupby("leaf_id"):
        leaf_rows.append({
            "leaf_id": leaf_id,
            "samples": len(group),
            "rmse_global": rmse(group[target_col], group["pred_global"]),
            "rmse_leaf": rmse(group[target_col], group["pred_leaf"]),
        })
    leaf_comparison_df = pd.DataFrame(leaf_rows).sort_values("samples", ascending=False)
    leaf_comparison_df["leaf_improvement_pct"] = (
        (leaf_comparison_df["rmse_global"] - leaf_comparison_df["rmse_leaf"])
        / leaf_comparison_df["rmse_global"] * 100
    )

    os.makedirs(results_dir, exist_ok=True)
    test_df.to_csv(os.path.join(results_dir, "test_all_predictions.csv"), index=False)
    leaf_comparison_df.to_csv(os.path.join(results_dir, "comparison_by_leaf.csv"), index=False)

    print(overall_results)
    return overall_results, leaf_comparison_df


if __name__ == "__main__":
    evaluate()
