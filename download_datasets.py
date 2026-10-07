"""Data-Driven model: one AutoGluon predictor per CART leaf."""
import os
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset

TARGET = "target"
SAVE_ROOT = "models/per_leaf_medium_notimelimit"


def train(train_csv="data/train_df_with_leaf.csv", target_col=TARGET, save_root=SAVE_ROOT):
    train_df = pd.read_csv(train_csv)
    leaf_models = {}
    for leaf_id, df_leaf in train_df.groupby("leaf_id"):
        X_train_leaf = df_leaf.drop(columns=["leaf_id"])
        predictor = TabularPredictor(
            label=target_col, eval_metric="rmse", problem_type="regression",
            path=os.path.join(save_root, f"leaf_{leaf_id}"),
        ).fit(train_data=TabularDataset(X_train_leaf), presets="medium", auto_stack=False)
        leaf_models[leaf_id] = predictor
    return leaf_models


def load(save_root=SAVE_ROOT, train_df=None, train_csv="data/train_df_with_leaf.csv"):
    if train_df is None:
        train_df = pd.read_csv(train_csv)
    leaf_id_list = list(train_df["leaf_id"].unique())
    return {leaf_id: TabularPredictor.load(os.path.join(save_root, f"leaf_{leaf_id}")) for leaf_id in leaf_id_list}


if __name__ == "__main__":
    train()
