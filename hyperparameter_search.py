"""Per-dataset hyperparameter search for the Data-Driven model.

Searches min_samples_leaf, max_leaf_nodes and the leaf-count selection rule
(1-SE vs. plain minimum). Uses only the training set, which is split into an
inner training part and a validation part; the test set is never touched.

  Stage 1: CART-only k-fold CV on the inner part, for every min_samples_leaf
           in the grid. Each selection rule picks a leaf count from the
           resulting RMSE curve.
  Stage 2: the top candidates are scored end-to-end: tree fit on the inner
           part, a time-limited AutoGluon model per leaf, RMSE on validation.
"""
import os
import shutil

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import cart
from cart_auto_tuning import LEAF_GRID, auto_select_leaf_count, is_dataset_viable_for_cart_leaf

RNG_SEED = 42
VAL_SIZE = 0.2
MIN_SAMPLES_LEAF_GRID = (100, 200, 300, 500, 800)
SELECTION_RULES = ("1se", "min")
TOP_K = 3
SEARCH_TIME_LIMIT = 60  # seconds per AutoGluon model in stage 2


def pick_leaf_count(cv_results, rule, se_multiplier=1.0):
    """Picks (leaves, mean_rmse, se) from a CV curve with the given rule."""
    best = min(cv_results, key=lambda r: r[1])
    if rule == "min":
        return best
    threshold = best[1] + se_multiplier * best[2]
    return min((r for r in cv_results if r[1] <= threshold), key=lambda r: r[0])


def stage1_cart_cv(X_inner, y_inner, msl_grid=MIN_SAMPLES_LEAF_GRID,
                   leaf_grid=LEAF_GRID, rng_seed=RNG_SEED):
    rows = []
    for msl in msl_grid:
        if not is_dataset_viable_for_cart_leaf(len(X_inner), msl):
            print(f"[search] skip min_samples_leaf={msl}: too few rows")
            continue
        rng = np.random.default_rng(rng_seed)
        try:
            _, cv_results = auto_select_leaf_count(
                X_inner, y_inner, rng, rng_seed, min_samples_leaf=msl, leaf_grid=leaf_grid,
            )
        except ValueError as err:
            print(f"[search] skip min_samples_leaf={msl}: {err}")
            continue
        for rule in SELECTION_RULES:
            leaves, cv_rmse, cv_se = pick_leaf_count(cv_results, rule)
            rows.append({
                "min_samples_leaf": msl, "rule": rule, "max_leaf_nodes": leaves,
                "cv_rmse": cv_rmse, "cv_se": cv_se,
            })
    return pd.DataFrame(rows)


def shortlist(stage1_df, top_k=TOP_K):
    """Takes the best candidates from each rule in turn, skipping duplicates."""
    ranked = {rule: g.sort_values("cv_rmse").to_dict("records")
              for rule, g in stage1_df.groupby("rule")}
    picked, seen, depth = [], set(), 0
    while len(picked) < top_k and any(depth < len(v) for v in ranked.values()):
        for rule in SELECTION_RULES:
            records = ranked.get(rule, [])
            if depth >= len(records):
                continue
            rec = records[depth]
            key = (rec["min_samples_leaf"], rec["max_leaf_nodes"])
            if key in seen:
                continue
            seen.add(key)
            picked.append(rec)
            if len(picked) == top_k:
                break
        depth += 1
    return pd.DataFrame(picked)


def score_end_to_end(inner_df, val_df, target_col, min_samples_leaf, max_leaf_nodes,
                     work_dir, time_limit=SEARCH_TIME_LIMIT, rng_seed=RNG_SEED):
    """Validation RMSE of the full Data-Driven model for one configuration."""
    X_inner, y_inner, X_val, _, features = cart.prepare_features(inner_df, val_df, target_col)
    tree = cart.fit_final_tree(X_inner, y_inner, features, max_leaf_nodes, min_samples_leaf, rng_seed)
    inner_leaf = tree.apply(X_inner)
    val_leaf = tree.apply(X_val)

    preds = np.full(len(val_df), np.nan)
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        for leaf_id in np.unique(inner_leaf):
            predictor = TabularPredictor(
                label=target_col, eval_metric="rmse", problem_type="regression",
                path=os.path.join(work_dir, f"leaf_{leaf_id}"), verbosity=0,
            ).fit(
                train_data=TabularDataset(inner_df[inner_leaf == leaf_id]),
                presets="medium", auto_stack=False, time_limit=time_limit,
            )
            val_mask = val_leaf == leaf_id
            if val_mask.any():
                X_leaf = val_df[val_mask].drop(columns=[target_col])
                preds[val_mask] = predictor.predict(TabularDataset(X_leaf)).values
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return float(np.sqrt(mean_squared_error(val_df[target_col], preds)))


def search(train_df, target_col="target", work_root="models/_search_tmp", log_path=None,
           msl_grid=MIN_SAMPLES_LEAF_GRID, leaf_grid=LEAF_GRID, top_k=TOP_K,
           time_limit=SEARCH_TIME_LIMIT, rng_seed=RNG_SEED):
    """Returns the best {min_samples_leaf, max_leaf_nodes, rule, val_rmse}, or None."""
    inner_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=rng_seed)
    X_inner, y_inner, _, _, _ = cart.prepare_features(inner_df, val_df, target_col)

    stage1 = stage1_cart_cv(X_inner, y_inner, msl_grid, leaf_grid, rng_seed)
    if stage1.empty:
        return None

    candidates = shortlist(stage1, top_k)
    print(f"[search] stage 2: scoring {len(candidates)} candidate(s) end-to-end")
    scores = []
    for i, cand in enumerate(candidates.to_dict("records")):
        val_rmse = score_end_to_end(
            inner_df, val_df, target_col,
            min_samples_leaf=int(cand["min_samples_leaf"]),
            max_leaf_nodes=int(cand["max_leaf_nodes"]),
            work_dir=os.path.join(work_root, f"cand_{i}"),
            time_limit=time_limit, rng_seed=rng_seed,
        )
        print(f"[search] min_samples_leaf={cand['min_samples_leaf']}, "
              f"leaves={cand['max_leaf_nodes']} ({cand['rule']}): val RMSE = {val_rmse:.6g}")
        scores.append(val_rmse)
    candidates["val_rmse"] = scores

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log = stage1.merge(
            candidates[["min_samples_leaf", "max_leaf_nodes", "val_rmse"]],
            on=["min_samples_leaf", "max_leaf_nodes"], how="left",
        )
        log.to_csv(log_path, index=False)

    best = candidates.loc[candidates["val_rmse"].idxmin()]
    chosen = {
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_leaf_nodes": int(best["max_leaf_nodes"]),
        "rule": best["rule"],
        "val_rmse": float(best["val_rmse"]),
    }
    print(f"[search] chosen: {chosen}")
    return chosen
