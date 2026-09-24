"""Checks each CTR-23 dataset's feature space for genuine heterogeneity
via unsupervised clustering, independent of the CART tree. If a dataset's
features form well-separated, non-trivial groups AND those groups differ
meaningfully in target value, that's evidence of real, exploitable
heterogeneity.

HDBSCAN (a) picks its own number of clusters from the data
and (b) labels points that don't belong to any sufficiently dense group as
noise (label -1) rather than forcing them into one, so a dataset with no
real structure can come back with zero clusters found.

For each dataset with data/<name>/raw.csv already downloaded:
  1. Fits HDBSCAN once (no k grid needed -- it finds its own cluster
     count). Points labeled -1 (noise) are excluded from all downstream
     metrics; noise_fraction_pct reports how much of the data that was.
  2. If fewer than 2 real clusters are found, the dataset is marked
     has_heterogeneity=False and nothing further is computed.
  3. If >=2 clusters are found but more than MAX_NOISE_PCT of rows are
     unassigned noise, the dataset is still scored during the run but has_heterogeneity
     is False and it is dropped.

"""
import argparse
import os
import traceback

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.stats import kruskal

from cart import prepare_features
from cart_auto_tuning import compute_min_samples_leaf

TARGET_COL = "target"
DATA_ROOT = "data"
HDBSCAN_SAMPLE_CAP = 60_000
RNG_SEED = 42
MAX_NOISE_PCT = 50.0

CTR23_DATASETS = {
    "Moneyball": 41021,
    "QSAR_fish_toxicity": 44970,
    "abalone": 44956,
    "airfoil_self_noise": 44957,
    "auction_verification": 44958,
    "brazilian_houses": 44990,
    "california_housing": 44977,
    "cars": 44994,
    "concrete_compressive_strength": 44959,
    "cps88wages": 44984,
    "cpu_activity": 44978,
    "diamonds": 44979,
    "energy_efficiency": 44960,
    "fifa": 45012,
    "fps_benchmark": 44992,
    "geographical_origin_of_music": 44965,
    "grid_stability": 44973,
    "health_insurance": 44993,
    "kin8nm": 44980,
    "kings_county": 44989,
    "miami_housing": 44983,
    "naval_propulsion_plant": 44969,
    "physiochemical_protein": 44963,
    "pumadyn32nh": 44981,
    "red_wine": 44972,
    "sarcos": 44976,
    "socmob": 44987,
    "solar_flare": 44966,
    "space_ga": 45402,
    "superconductivity": 44964,
    "video_transcoding": 44974,
    "wave_energy": 44975,
    "white_wine": 44971,
}


def target_variance_reduction(y, labels):
    """% of global (non-noise) target variance explained by cluster membership."""
    global_var = np.var(y)
    if global_var == 0:
        return np.nan
    within = sum(
        np.sum(labels == c) * np.var(y[labels == c])
        for c in np.unique(labels)
    ) / len(y)
    return (1 - within / global_var) * 100


def compute_one(name, data_root=DATA_ROOT, rng_seed=RNG_SEED,
                 sample_cap=HDBSCAN_SAMPLE_CAP, max_noise_pct=MAX_NOISE_PCT):
    raw_path = os.path.join(data_root, name, "raw.csv")
    if not os.path.isfile(raw_path):
        return {"dataset": name, "status": f"missing {raw_path}"}

    df = pd.read_csv(raw_path)
    if TARGET_COL not in df.columns:
        return {"dataset": name, "status": f"no '{TARGET_COL}' column"}

    X, y, _, _, _ = prepare_features(df, df, target_col=TARGET_COL)
    n_rows = len(X)
    min_cluster_size = compute_min_samples_leaf(n_rows)
    if n_rows < 2 * min_cluster_size:
        return {"dataset": name, "status": f"too few rows ({n_rows}) for min_cluster_size={min_cluster_size}"}

    rng = np.random.default_rng(rng_seed)
    if n_rows > sample_cap:
        idx = rng.choice(n_rows, sample_cap, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]
        n_rows = sample_cap
        print(f"{name}: subsampled to {sample_cap} rows for HDBSCAN tractability")

    X_scaled = StandardScaler().fit_transform(X)
    y_arr = y.to_numpy()

    labels = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(X_scaled)
    is_noise = labels == -1
    noise_fraction = is_noise.mean()
    found_labels = labels[~is_noise]
    n_clusters_found = len(np.unique(found_labels))
    print(f"{name} | min_cluster_size={min_cluster_size} | clusters_found={n_clusters_found} "
          f"| noise_fraction={noise_fraction:.2%}")

    base_row = {
        "dataset": name,
        "status": "ok",
        "n_rows": n_rows,
        "min_cluster_size": min_cluster_size,
        "n_clusters_found": n_clusters_found,
        "noise_fraction_pct": noise_fraction * 100,
    }

    if n_clusters_found < 2:
        return {
            **base_row,
            "has_heterogeneity": False,
            "target_variance_reduction_pct": np.nan,
            "kruskal_pvalue": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    X_clustered = X_scaled[~is_noise]
    y_clustered = y_arr[~is_noise]
    var_red = target_variance_reduction(y_clustered, found_labels)
    groups = [y_clustered[found_labels == c] for c in np.unique(found_labels)]
    kruskal_p = kruskal(*groups).pvalue
    ch_score = calinski_harabasz_score(X_clustered, found_labels)
    db_score = davies_bouldin_score(X_clustered, found_labels)

    # Real cluster structure was found, but if it's mostly noise, it's a
    # small dense pocket rather than the dataset splitting into distinct
    # subpopulations as a whole.
    has_heterogeneity = (noise_fraction * 100) <= max_noise_pct
    if not has_heterogeneity:
        print(f"{name}: excluded -- noise_fraction={noise_fraction:.2%} exceeds "
              f"max_noise_pct={max_noise_pct:.0f}%")

    return {
        **base_row,
        "has_heterogeneity": has_heterogeneity,
        "target_variance_reduction_pct": var_red,
        "kruskal_pvalue": kruskal_p,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score,
    }


def main(datasets=None, data_root=DATA_ROOT, max_noise_pct=MAX_NOISE_PCT):
    names = datasets or list(CTR23_DATASETS.keys())
    rows = []
    for name in names:
        print(f"=== {name} ===")
        try:
            row = compute_one(name, data_root=data_root, max_noise_pct=max_noise_pct)
        except Exception as e:
            row = {"dataset": name, "status": f"ERROR: {e}"}
            traceback.print_exc()
        print(row)
        rows.append(row)

    df = pd.DataFrame(rows)

    ok = df[df["status"] == "ok"]
    has_het = ok[ok["has_heterogeneity"] == True].sort_values("calinski_harabasz", ascending=False)
    no_het = ok[ok["has_heterogeneity"] == False]

    print(f"\n--- {len(has_het)}/{len(ok)} datasets pass the heterogeneity check "
          f"(>=2 real HDBSCAN clusters AND noise_fraction_pct <= {max_noise_pct:.0f}%) ---")
    if not has_het.empty:
        print(has_het[["dataset", "n_rows", "min_cluster_size", "n_clusters_found",
                        "noise_fraction_pct", "calinski_harabasz", "davies_bouldin",
                        "target_variance_reduction_pct", "kruskal_pvalue"]].to_string(index=False))
    if not no_het.empty:
        print(f"\nExcluded (no structure, or too noisy -- not written to disk): "
              f"{', '.join(no_het['dataset'])}")

    # Only the datasets that pass the heterogeneity check remain.
    os.makedirs("results", exist_ok=True)
    has_het.to_csv("results/heterogeneity_check.csv", index=False)
    print(f"\nWrote {len(has_het)} dataset(s) passing the heterogeneity check "
          f"to results/heterogeneity_check.csv")

    missing = df[df["status"] != "ok"]
    if not missing.empty:
        print(f"\n{len(missing)} dataset(s) skipped:")
        print(missing[["dataset", "status"]].to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="subset of dataset names to check (default: all 33)")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--max-noise-pct", type=float, default=MAX_NOISE_PCT,
                         help="datasets with real clusters but more than this pct noise are excluded (default 50.0)")
    args = parser.parse_args()
    main(datasets=args.datasets, data_root=args.data_root, max_noise_pct=args.max_noise_pct)
