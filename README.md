# Partitioning Strategies for Improving AutoML Performance on Heterogeneous Virtual Metrology Data: Public Demo (CTR-23)

Reproduces the paper's public demonstration: Global vs. Data-Driven
RMSE on datasets from the OpenML CTR-23 regression benchmark suite
(suite 353) that pass a heterogeneity check -- currently 6 of the
suite's 33 datasets:

```
abalone
brazilian_houses
cps88wages
diamonds
naval_propulsion_plant
video_transcoding
```

## Approach

For each dataset, two models are trained on the same 80/20 train/test
split and compared on held-out RMSE:

- **Global**: one AutoGluon predictor trained on the full training set.
- **Data-Driven**: a `DecisionTreeRegressor` (CART) partitions the
  training data into leaves (leaf count chosen by cross-validated RMSE
  on the training split), and a separate AutoGluon predictor is
  trained per leaf.

Before either model is trained, each candidate dataset is screened for
genuine heterogeneity via HDBSCAN clustering on its feature space
(independent of the CART tree, and without looking at test RMSE): a
dataset only proceeds to the model comparison if it has real cluster
structure (>=2 clusters) with no more than 50% of rows falling outside
any cluster ("noise").

## Files

- `download_all_ctr23.py` -- downloads all 33 CTR-23 datasets from
  suite 353 into `data/<dataset>/raw.csv`.
- `heterogeneity_check.py` -- runs the HDBSCAN heterogeneity
  check on each downloaded dataset; writes the datasets that pass to
  `results/heterogeneity_check.csv`.
- `data.py` -- train/test split (80/20, seed 42).
- `cart_auto_tuning.py` -- cross-validated selection of CART's leaf
  count.
- `cart.py` -- fits the CART tree, assigns each row a `leaf_id`, and
  plots the tree structure.
- `train_global_model.py` -- trains the Global AutoGluon model.
- `train_cart_model.py` -- trains one AutoGluon model per leaf.
- `evaluate.py` -- scores both models on the test set; writes per-row
  and per-leaf RMSE breakdowns.
- `run_pipeline.py` -- for each dataset that currently passes
  the heterogeneity check (read from `results/heterogeneity_check.csv`):
  split -> CART -> train Global + per-leaf AutoGluon models ->
  evaluate; writes `clustered_selection_results.csv`.

## Running

```bash
pip install -r requirements.txt
python download_all_ctr23.py
python heterogeneity_check.py
python run_pipeline.py
```
## Resulst 
