"""PartitionedRegressor: CART-partitioned AutoGluon regression that tunes
itself.

Usage:
    from partitioned_regressor import PartitionedRegressor

    # public data: random splits
    model = PartitionedRegressor(label="target", path="models/my_model")
    # fab data: chronological splits
    model = PartitionedRegressor(label="target", path="models/my_model",
                                 split="chronological", time_col="timestamp")

    model.fit(train_df)              # searches, trains the final model, saves it
    preds = model.predict(test_df)
    print(model.evaluate(test_df))   # {"rmse": ...}
    print(model.summary())           # what the search chose
    print(model.search_results_)     # every candidate and its validation score

    model = PartitionedRegressor.load("models/my_model")

How fit() works (training data only, the test set is never touched):
  1. Candidates (cheap): CART-only k-fold CV over leaf counts for every
     min_samples_leaf in the grid; the 1-SE rule picks one leaf count per
     curve. The best few by CV RMSE are kept, plus the paper's default
     configuration.
  2. Scoring (end-to-end): every candidate is trained with AutoGluon per leaf
     and scored on validation data. Small datasets use k-fold CV, large ones
     one holdout split. Folds are random or chronological (`split`).
  3. Selection (1-SE rule): the simplest candidate whose validation RMSE is
     within one standard error of the best wins: fewer leaves first, then
     larger min_samples_leaf.
  4. The winner is refit on all training data with the final AutoGluon
     settings and saved to `path`.

min_samples_leaf is always meant relative to the full training set; when a
tree is fit on fewer rows (a CV fold), it is scaled down proportionally so the
partition stays comparable.
"""
import os
import pickle
import shutil

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor

MODEL_FILE = "partitioned_regressor.pkl"
CART_SAMPLE_CAP = 200_000
N_BOOTSTRAP = 200


def rmse(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def default_min_samples_leaf(n_train):
    """The paper's formula: 5% of the training rows, clipped to [300, 800]."""
    return int(np.clip(round(0.05 * n_train), 300, 800))


def _pick_1se(curve):
    """Smallest leaf count within one SE of the best: (leaves, mean_rmse, se)."""
    best = min(curve, key=lambda r: r[1])
    return min((r for r in curve if r[1] <= best[1] + best[2]), key=lambda r: r[0])


class _TreeFeatures:
    """Numeric, complete feature matrix for CART: drops ID-like and datetime
    columns, one-hot encodes categoricals, adds a missingness indicator per
    column with gaps, and imputes medians."""

    def __init__(self, max_unique_ratio=0.5):
        self.max_unique_ratio = max_unique_ratio

    def fit(self, X):
        n = len(X)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        id_like = [c for c in cat_cols if X[c].nunique(dropna=True) / n > self.max_unique_ratio]
        time_like = X.select_dtypes(include=["datetime", "datetimetz", "timedelta"]).columns.tolist()
        self.dropped_cols_ = id_like + time_like
        self.cat_cols_ = [c for c in cat_cols if c not in id_like]
        self.missing_cols_ = [c for c in X.columns if c not in self.dropped_cols_ and X[c].isna().any()]
        encoded = self._encode(X)
        self.columns_ = encoded.columns.tolist()
        self.medians_ = encoded.median(numeric_only=True)
        return self

    def _encode(self, X):
        X = X.drop(columns=self.dropped_cols_)
        indicators = X[self.missing_cols_].isna().astype("int8").add_suffix("__missing")
        X = pd.get_dummies(X, columns=self.cat_cols_, dtype="int8")
        return pd.concat([X, indicators], axis=1)

    def transform(self, X):
        encoded = self._encode(X).reindex(columns=self.columns_, fill_value=0)
        return encoded.fillna(self.medians_).fillna(0)


class PartitionedRegressor:
    def __init__(self, label, path="partitioned_model", split="random", time_col=None,
                 drop_time_col=True, search=True,
                 min_samples_leaf_grid=(200, 300, 500, 800),
                 leaf_cap=15, max_leaf_cap=60, top_k=3,
                 small_data_rows=5000, n_folds=3, val_fraction=0.2,
                 search_time_limit=60, final_fit_kwargs=None,
                 random_state=42, verbosity=1):
        """
        label:                 target column.
        path:                  folder for the final models.
        split:                 'random' (public data) or 'chronological'
                               (fab data, needs time_col). Used for every
                               CV fold and holdout in the search.
        time_col:              timestamp column for chronological splits.
        drop_time_col:         keep time_col out of the model features.
        search:                False = skip the search and use the paper's
                               default configuration.
        min_samples_leaf_grid: values tried, relative to the full training set.
        leaf_cap:              starting leaf-count cap; doubled (up to
                               max_leaf_cap) while the best count sits on it.
        top_k:                 searched candidates scored end-to-end, on top
                               of the default candidate.
        small_data_rows:       below this many rows, score with n_folds-fold CV;
                               above it, with one holdout split.
        search_time_limit:     seconds per AutoGluon model during the search.
        final_fit_kwargs:      kwargs for the final AutoGluon fits (default:
                               presets='medium', auto_stack=False, no limit).
        """
        if split not in ("random", "chronological"):
            raise ValueError("split must be 'random' or 'chronological'")
        if split == "chronological" and time_col is None:
            raise ValueError("split='chronological' needs time_col")
        self.label = label
        self.path = path
        self.split = split
        self.time_col = time_col
        self.drop_time_col = drop_time_col
        self.search = search
        self.min_samples_leaf_grid = tuple(min_samples_leaf_grid)
        self.leaf_cap = leaf_cap
        self.max_leaf_cap = max_leaf_cap
        self.top_k = top_k
        self.small_data_rows = small_data_rows
        self.n_folds = n_folds
        self.val_fraction = val_fraction
        self.search_time_limit = search_time_limit
        self.final_fit_kwargs = final_fit_kwargs or {"presets": "medium", "auto_stack": False}
        self.random_state = random_state
        self.verbosity = verbosity

    # ------------------------------------------------------------------ public

    def fit(self, train_data):
        data = train_data.reset_index(drop=True)
        n = len(data)
        self.feature_cols_ = [
            c for c in data.columns
            if c != self.label and not (self.drop_time_col and c == self.time_col)
        ]
        self.n_train_ = n
        self._log(f"Fitting on {n} rows, {len(self.feature_cols_)} features, {self.split} splits")

        rank = self._time_rank(data)
        X, y = self._tree_matrix(data), data[self.label]

        if self.search:
            candidates = self._generate_candidates(X, y, rank)
            self.search_results_ = self._score_candidates(candidates, data, rank)
            self.chosen_ = self._select(self.search_results_)
        else:
            self.search_results_ = None
            msl = default_min_samples_leaf(n)
            curve = self._cart_curve(X, y, msl, rank) if n >= 2 * msl else []
            if not curve:
                raise ValueError(f"Too few rows ({n}) for the default min_samples_leaf={msl}.")
            self.chosen_ = self._default_candidate(n, {msl: curve})

        self._log(f"Chosen: {self._describe(self.chosen_)}. Training final model...")
        final_dir = os.path.join(self.path, "final")
        shutil.rmtree(final_dir, ignore_errors=True)
        self._final = self._fit_candidate(
            self.chosen_, data, final_dir, self.final_fit_kwargs,
            ag_verbosity=2 if self.verbosity >= 2 else 0,
        )
        self.save()
        self._log(f"Done: {self.summary()}")
        return self

    def predict(self, data):
        return pd.Series(self._predict_fitted(self._final, data), index=data.index, name=self.label)

    def evaluate(self, data):
        return {"rmse": rmse(data[self.label], self.predict(data))}

    def leaf_of(self, data):
        """Leaf id per row."""
        return self._final["tree"].apply(self._final["features"].transform(data[self.feature_cols_]))

    def summary(self):
        return (f"CART partition with {self._final['tree'].get_n_leaves()} leaves, "
                f"min_samples_leaf={self.chosen_['min_samples_leaf']} ({self.chosen_['source']})")

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        state = dict(self.__dict__)
        final = dict(self._final)
        final["predictors"] = {k: os.path.relpath(p.path, self.path) for k, p in final["predictors"].items()}
        state["_final"] = final
        with open(os.path.join(self.path, MODEL_FILE), "wb") as f:
            pickle.dump(state, f)
        return self.path

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, MODEL_FILE), "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj.path = path
        obj._final["predictors"] = {
            k: TabularPredictor.load(os.path.join(path, rel)) for k, rel in obj._final["predictors"].items()
        }
        return obj

    # ---------------------------------------------------------------- splits

    def _time_rank(self, data):
        """Position of each row in time order (None for random splits)."""
        if self.split != "chronological":
            return None
        t = data[self.time_col]
        if not (pd.api.types.is_numeric_dtype(t) or pd.api.types.is_datetime64_any_dtype(t)):
            t = pd.to_datetime(t, errors="coerce")
        order = np.argsort(t.to_numpy(), kind="stable")
        rank = np.empty(len(data), dtype=int)
        rank[order] = np.arange(len(data))
        return rank

    def _cv_folds(self, n, rank):
        """k folds: shuffled KFold, or expanding-window folds in time order."""
        if rank is None:
            return list(KFold(self.n_folds, shuffle=True, random_state=self.random_state).split(np.arange(n)))
        order = np.argsort(rank, kind="stable")
        return [(order[tr], order[va]) for tr, va in TimeSeriesSplit(self.n_folds).split(order)]

    def _validation_splits(self, n, rank):
        """Stage-2 splits: k folds for small data, one holdout for large data."""
        if n < self.small_data_rows:
            return self._cv_folds(n, rank)
        if rank is None:
            tr, va = train_test_split(np.arange(n), test_size=self.val_fraction, random_state=self.random_state)
            return [(tr, va)]
        order = np.argsort(rank, kind="stable")
        n_val = max(1, int(round(n * self.val_fraction)))
        return [(order[:-n_val], order[-n_val:])]

    # ------------------------------------------------------------ candidates

    def _default_candidate(self, n, curves):
        msl = default_min_samples_leaf(n)
        if msl not in curves:
            return None
        leaves, cv_rmse, _ = _pick_1se(curves[msl])
        return {"source": "default", "min_samples_leaf": msl,
                "max_leaf_nodes": leaves, "cart_cv_rmse": cv_rmse}

    def _generate_candidates(self, X, y, rank):
        n = len(X)
        curves = {}
        for msl in sorted(set(self.min_samples_leaf_grid) | {default_min_samples_leaf(n)}):
            if n >= 2 * msl:
                curve = self._cart_curve(X, y, msl, rank)
                if curve:
                    curves[msl] = curve
        if not curves:
            raise ValueError(f"Too few rows ({n}) to partition with min_samples_leaf >= "
                             f"{min(self.min_samples_leaf_grid)}.")

        candidates = []
        default = self._default_candidate(n, curves)
        if default:
            candidates.append(default)

        pool = []
        for msl, curve in curves.items():
            leaves, cv_rmse, _ = _pick_1se(curve)
            pool.append({"source": "search", "min_samples_leaf": msl,
                         "max_leaf_nodes": leaves, "cart_cv_rmse": cv_rmse})

        seen = {(c["min_samples_leaf"], c["max_leaf_nodes"]) for c in candidates}
        added = 0
        for rec in sorted(pool, key=lambda r: r["cart_cv_rmse"]):
            key = (rec["min_samples_leaf"], rec["max_leaf_nodes"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(rec)
            added += 1
            if added >= self.top_k:
                break

        self._log(f"Stage 1: {len(curves)} min_samples_leaf value(s) screened, "
                  f"{len(candidates)} candidate(s) go to end-to-end scoring")
        return candidates

    def _cart_curve(self, X, y, min_samples_leaf, rank):
        """CART-only k-fold CV: [(leaves, mean_rmse, se), ...]. The leaf cap
        doubles (up to max_leaf_cap) while the best leaf count sits on it."""
        n_full = len(X)
        if n_full > CART_SAMPLE_CAP:
            idx = np.sort(np.random.default_rng(self.random_state).choice(n_full, CART_SAMPLE_CAP, replace=False))
            X, y = X.iloc[idx], y.iloc[idx]
            rank = None if rank is None else rank[idx]
        folds = self._cv_folds(len(X), rank)
        fold_msl = [max(1, int(round(min_samples_leaf * len(tr) / n_full))) for tr, _ in folds]

        curve, leaves, cap = [], 2, self.leaf_cap
        while True:
            while leaves <= cap and all(leaves * m <= len(tr) for m, (tr, _) in zip(fold_msl, folds)):
                scores = []
                for m, (tr, va) in zip(fold_msl, folds):
                    tree = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=m,
                                                 random_state=self.random_state)
                    tree.fit(X.iloc[tr], y.iloc[tr])
                    scores.append(rmse(y.iloc[va], tree.predict(X.iloc[va])))
                curve.append((leaves, float(np.mean(scores)),
                              float(np.std(scores, ddof=1) / np.sqrt(len(scores)))))
                leaves += 1
            hit_cap = bool(curve) and leaves > cap and min(curve, key=lambda r: r[1])[0] == cap
            if not hit_cap or cap >= self.max_leaf_cap:
                return curve
            cap = min(2 * cap, self.max_leaf_cap)

    # --------------------------------------------------------------- scoring

    def _score_candidates(self, candidates, data, rank):
        splits = self._validation_splits(len(data), rank)
        scheme = f"{len(splits)}-fold CV" if len(splits) > 1 else "holdout"
        self._log(f"Stage 2: scoring end-to-end with {self.split} {scheme}")
        search_kwargs = {"presets": "medium", "auto_stack": False, "time_limit": self.search_time_limit}
        search_dir = os.path.join(self.path, "_search")
        rows = []
        try:
            for i, cand in enumerate(candidates):
                fold_rmses, sq_errors = [], []
                for j, (tr, va) in enumerate(splits):
                    work_dir = os.path.join(search_dir, f"cand{i}_fold{j}")
                    train_part, val_part = data.iloc[tr], data.iloc[va]
                    fitted = self._fit_candidate(cand, train_part, work_dir, search_kwargs,
                                                 n_reference=len(data))
                    preds = self._predict_fitted(fitted, val_part)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    err = val_part[self.label].to_numpy(dtype=float) - preds
                    fold_rmses.append(float(np.sqrt(np.mean(err ** 2))))
                    sq_errors.append(err ** 2)
                if len(splits) > 1:
                    val_rmse = float(np.mean(fold_rmses))
                    val_se = float(np.std(fold_rmses, ddof=1) / np.sqrt(len(fold_rmses)))
                else:
                    val_rmse, val_se = fold_rmses[0], self._bootstrap_se(sq_errors[0])
                rows.append({**cand, "val_rmse": val_rmse, "val_se": val_se})
                self._log(f"  {self._describe(cand)}: val RMSE {val_rmse:.6g} +/- {val_se:.3g}")
        finally:
            shutil.rmtree(search_dir, ignore_errors=True)
        return pd.DataFrame(rows)

    def _bootstrap_se(self, sq_err):
        rng = np.random.default_rng(self.random_state)
        n = len(sq_err)
        boots = [np.sqrt(sq_err[rng.integers(0, n, n)].mean()) for _ in range(N_BOOTSTRAP)]
        return float(np.std(boots, ddof=1))

    def _select(self, results):
        """1-SE rule: simplest candidate within one SE of the best."""
        results = results.copy()
        best = results.loc[results["val_rmse"].idxmin()]
        pool = results[results["val_rmse"] <= best["val_rmse"] + best["val_se"]]
        winner = pool.sort_values(["max_leaf_nodes", "min_samples_leaf"], ascending=[True, False]).index[0]
        results["chosen"] = results.index == winner
        self.search_results_ = results
        row = results.loc[winner]
        return {"source": row["source"], "min_samples_leaf": int(row["min_samples_leaf"]),
                "max_leaf_nodes": int(row["max_leaf_nodes"])}

    # ------------------------------------------------------- fit / predict

    def _tree_matrix(self, data):
        X = data[self.feature_cols_]
        return _TreeFeatures().fit(X).transform(X)

    def _train_ag(self, df, path, fit_kwargs, ag_verbosity=0):
        shutil.rmtree(path, ignore_errors=True)
        return TabularPredictor(
            label=self.label, eval_metric="rmse", problem_type="regression",
            path=path, verbosity=ag_verbosity,
        ).fit(df, **fit_kwargs)

    def _fit_candidate(self, cand, data, work_dir, fit_kwargs, n_reference=None, ag_verbosity=0):
        """Trains one candidate on `data`. With n_reference, min_samples_leaf
        is rescaled from n_reference rows to len(data) rows."""
        model_data = data[self.feature_cols_ + [self.label]]
        msl = cand["min_samples_leaf"]
        if n_reference:
            msl = max(1, int(round(msl * len(data) / n_reference)))
        features = _TreeFeatures().fit(data[self.feature_cols_])
        X = features.transform(data[self.feature_cols_])
        tree = DecisionTreeRegressor(max_leaf_nodes=int(cand["max_leaf_nodes"]), min_samples_leaf=msl,
                                     random_state=self.random_state).fit(X, data[self.label])
        leaf_ids = tree.apply(X)
        predictors = {}
        for leaf_id in np.unique(leaf_ids):
            predictors[int(leaf_id)] = self._train_ag(
                model_data[leaf_ids == leaf_id], os.path.join(work_dir, f"leaf_{leaf_id}"),
                fit_kwargs, ag_verbosity,
            )
        return {"features": features, "tree": tree, "predictors": predictors}

    def _predict_fitted(self, fitted, data):
        X_model = data[self.feature_cols_]
        leaf_ids = fitted["tree"].apply(fitted["features"].transform(X_model))
        preds = np.full(len(data), np.nan)
        for leaf_id, predictor in fitted["predictors"].items():
            mask = leaf_ids == leaf_id
            if mask.any():
                preds[mask] = predictor.predict(X_model[mask]).to_numpy(dtype=float)
        return preds

    # --------------------------------------------------------------- misc

    def _describe(self, cand):
        return f"{cand['source']}: min_samples_leaf={cand['min_samples_leaf']}, leaves={cand['max_leaf_nodes']}"

    def _log(self, msg):
        if self.verbosity:
            print(f"[PartitionedRegressor] {msg}")
