"""Cross-validated selection of CART's min_samples_leaf and leaf count."""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

MIN_SAMPLES_LEAF_FRACTION = 0.05
MIN_SAMPLES_LEAF_FLOOR = 300
MIN_SAMPLES_LEAF_CEILING = 800
MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF_CEILING
LEAF_GRID = range(2, 11)
SE_MULTIPLIER = 1.0


def compute_min_samples_leaf(n_train, fraction=MIN_SAMPLES_LEAF_FRACTION,
                              floor=MIN_SAMPLES_LEAF_FLOOR, ceiling=MIN_SAMPLES_LEAF_CEILING):
    """fraction * n_train (5% by default), clipped to [floor, ceiling]."""
    return int(np.clip(round(fraction * n_train), floor, ceiling))


def is_dataset_viable_for_cart_leaf(n_train, min_samples_leaf=MIN_SAMPLES_LEAF):
    """True if n_train supports at least 2 leaves at min_samples_leaf."""
    return n_train >= 2 * min_samples_leaf


def auto_select_leaf_count(X_train, y_train, rng, rng_seed,
                            min_samples_leaf=MIN_SAMPLES_LEAF,
                            leaf_grid=LEAF_GRID, n_splits=3, sample_cap=200_000,
                            use_1se_rule=True, se_multiplier=SE_MULTIPLIER):
    """Selects max_leaf_nodes over leaf_grid via k-fold CV RMSE (1-SE rule by default)."""
    n_train = len(X_train)
    if not is_dataset_viable_for_cart_leaf(n_train, min_samples_leaf):
        raise ValueError(f"n_train={n_train} too small for min_samples_leaf={min_samples_leaf}.")

    sample_size = min(sample_cap, n_train)
    sample_idx = rng.choice(n_train, sample_size, replace=False)
    X_sample = X_train.iloc[sample_idx]
    y_sample = y_train.iloc[sample_idx]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    results = []
    for leaves in leaf_grid:
        if leaves * min_samples_leaf > n_train:
            break

        fold_scores = []
        for train_idx, val_idx in kf.split(X_sample):
            X_tr, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
            y_tr, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]
            tree = DecisionTreeRegressor(
                max_leaf_nodes=leaves, min_samples_leaf=min_samples_leaf, random_state=rng_seed
            )
            tree.fit(X_tr, y_tr)
            preds = tree.predict(X_val)
            fold_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        avg_rmse = np.mean(fold_scores)
        se_rmse = (np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores))
                   if len(fold_scores) > 1 else 0.0)
        results.append((leaves, avg_rmse, se_rmse))
        print(f"Leaves: {leaves} | CV RMSE: {avg_rmse:.5f} +/- {se_rmse:.5f}")

    if not results:
        raise ValueError(f"No leaf count in {list(leaf_grid)} achievable with these settings.")

    min_leaves, min_rmse, min_se = min(results, key=lambda r: r[1])

    if use_1se_rule:
        threshold = min_rmse + se_multiplier * min_se
        candidates = [r for r in results if r[1] <= threshold]
        best_leaf_count, best_rmse, best_se = min(candidates, key=lambda r: r[0])
    else:
        best_leaf_count, best_rmse, best_se = min_leaves, min_rmse, min_se

    print(f"Selected leaf count: {best_leaf_count} (CV RMSE = {best_rmse:.5f})")
    return best_leaf_count, results
