"""Global baseline: one AutoGluon predictor trained on the full training set."""
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset

TARGET = "target"
SAVE_PATH = "models/global_medium_notimelimit"


def train(train_csv="data/train_df.csv", target_col=TARGET, save_path=SAVE_PATH):
    train_df = pd.read_csv(train_csv)
    global_model = TabularPredictor(
        label=target_col, eval_metric="rmse", problem_type="regression", path=save_path,
    ).fit(train_data=TabularDataset(train_df), presets="medium", auto_stack=False)
    return global_model


def load(save_path=SAVE_PATH):
    return TabularPredictor.load(save_path)


if __name__ == "__main__":
    train()
