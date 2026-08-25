"""CART tree construction: partitions data into homogeneous leaves."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text

from cart_auto_tuning import (
    is_dataset_viable_for_cart_leaf,
    auto_select_leaf_count,
    compute_min_samples_leaf,
    MIN_SAMPLES_LEAF,
)

TARGET = "target"
RNG_SEED = 42


def prepare_features(train_df, test_df, target_col=TARGET, max_unique_ratio=0.5):
    tree_train = train_df.copy()
    tree_test = test_df.copy()

    categorical_cols = tree_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    n = len(tree_train)
    id_like_cols = [c for c in categorical_cols if tree_train[c].nunique(dropna=True) / n > max_unique_ratio]
    if id_like_cols:
        categorical_cols = [c for c in categorical_cols if c not in id_like_cols]
        tree_train = tree_train.drop(columns=id_like_cols)
        tree_test = tree_test.drop(columns=id_like_cols)

    tree_train = pd.get_dummies(tree_train, columns=categorical_cols, drop_first=False, dtype="int8")
    tree_test = pd.get_dummies(tree_test, columns=categorical_cols, drop_first=False, dtype="int8")
    tree_train, tree_test = tree_train.align(tree_test, join="left", axis=1, fill_value=0)

    cart_features = [c for c in tree_train.columns if c != target_col]
    X_train = tree_train[cart_features]
    y_train = tree_train[target_col]
    X_test = tree_test[cart_features]
    y_test = tree_test[target_col]

    if X_train.isna().any().any():
        medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

    return X_train, y_train, X_test, y_test, cart_features


def fit_final_tree(X_train, y_train, cart_features, best_leaf_count, best_min_leaf=MIN_SAMPLES_LEAF,
                    rng_seed=RNG_SEED):
    tree = DecisionTreeRegressor(
        max_leaf_nodes=best_leaf_count, min_samples_leaf=best_min_leaf, random_state=rng_seed
    )
    tree.fit(X_train, y_train)
    print("Leaves:", tree.get_n_leaves())
    print(export_text(tree, feature_names=cart_features, spacing=3, decimals=2, show_weights=True))
    return tree


def feature_importance(tree, cart_features, out_path="figures/cart_feature_importance.png",
                        title="Top 5 Features Used by CART Tree for Splitting"):
    feat_imp = pd.DataFrame({
        "feature": cart_features, "importance": tree.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp = feat_imp[feat_imp["importance"] > 0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    feat_imp.head(5).plot(kind="barh", x="feature", y="importance", ax=ax, legend=False, color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return feat_imp


def summarize_groups(frame, group_col, target_col=TARGET):
    rows = []
    for group_val, count in frame[group_col].value_counts().items():
        mask = frame[group_col] == group_val
        y_group = frame.loc[mask, target_col].values
        mean_ = np.mean(y_group)
        var_ = np.var(y_group)
        std_ = np.sqrt(var_)
        cv_ = std_ / mean_ if mean_ != 0 else np.nan
        rows.append({"group": group_val, "samples": count, "mean": mean_, "variance": var_, "std": std_, "cv": cv_})
    return pd.DataFrame(rows).sort_values("samples", ascending=False).reset_index(drop=True)


def run(train_df, test_df, target_col=TARGET, figures_dir="figures",
        min_samples_leaf=None, leaf_grid=None, rng_seed=RNG_SEED):
    """Prepares features, auto-tunes leaf count, fits the tree, attaches leaf_id."""
    X_train, y_train, X_test, y_test, cart_features = prepare_features(train_df, test_df, target_col)

    if min_samples_leaf is None:
        min_samples_leaf = compute_min_samples_leaf(len(X_train))

    if not is_dataset_viable_for_cart_leaf(len(X_train), min_samples_leaf):
        print(f"SKIP: only {len(X_train)} training rows, need at least {2 * min_samples_leaf}.")
        return None

    rng = np.random.default_rng(rng_seed)
    auto_select_kwargs = {"min_samples_leaf": min_samples_leaf}
    if leaf_grid is not None:
        auto_select_kwargs["leaf_grid"] = leaf_grid
    best_leaf_count, cv_results = auto_select_leaf_count(
        X_train, y_train, rng, rng_seed, **auto_select_kwargs
    )

    tree = fit_final_tree(X_train, y_train, cart_features, best_leaf_count, min_samples_leaf, rng_seed)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["leaf_id"] = tree.apply(X_train)
    test_df["leaf_id"] = tree.apply(X_test)

    feature_importance(tree, cart_features, out_path=os.path.join(figures_dir, "cart_feature_importance.png"))
    leaf_summary_train = summarize_groups(train_df, "leaf_id", target_col)

    return {
        "tree": tree,
        "train_df": train_df,
        "test_df": test_df,
        "best_leaf_count": best_leaf_count,
        "min_samples_leaf": min_samples_leaf,
        "cv_results": cv_results,
        "leaf_summary_train": leaf_summary_train,
    }
