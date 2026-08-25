"""Dataset loading and train/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

RNG_SEED = 42
TEST_SIZE = 0.2
TARGET = "target"


def split(csv_path="data/raw.csv", target_col=TARGET,
          train_path="data/train_df.csv", test_path="data/test_df.csv",
          test_size=TEST_SIZE, rng_seed=RNG_SEED):
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=rng_seed)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


if __name__ == "__main__":
    split()
