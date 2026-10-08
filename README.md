# Partitioning Strategies for Improving AutoML Performance on Heterogeneous Virtual Metrology Data: Public Demo

Reproduces the paper's public demonstration: Global vs. Data-Driven
RMSE on five public regression datasets from different fields:

```
naval_propulsion_plant
video_transcoding
auction_verification
socmob
airfoil_self_noise
```

These datasets are chosen to show cases where partitioning helps. They
illustrate the approach on data that can be shared.
## Approach

For each dataset, two models are trained on the same 80/20 train/test
split and compared on held-out RMSE:

- **Global**: one AutoGluon predictor trained on the full training set.
- **Data-Driven**: a `DecisionTreeRegressor` (CART) partitions the
  training data into leaves, and a separate AutoGluon predictor is
  trained per leaf.

## Files

- `download_datasets.py` -- downloads the five datasets from OpenML
  into `data/<dataset>/raw.csv`.
- `data.py` -- train/test split (80/20, seed 42).
- `cart_auto_tuning.py` -- cross-validated selection of CART's leaf
  count.
- `cart.py` -- fits the CART tree, assigns each row a `leaf_id`, and
  plots the tree structure.
- `train_global_model.py` -- trains the Global AutoGluon model.
- `train_cart_model.py` -- trains one AutoGluon model per leaf.
- `evaluate.py` -- scores both models on the test set; writes per-row
  and per-leaf RMSE breakdowns.
- `run_pipeline.py` -- for each of the five datasets:
  split -> CART -> train Global + per-leaf AutoGluon models ->
  evaluate; writes `demo_results.csv`.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for dependency and interpreter
  management. Install it once with `brew install uv` or
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- No separate Python install is needed: `pyproject.toml`/`.python-version`
  pin the exact interpreter (3.12.7), and `uv sync` downloads it
  automatically if it isn't already on your machine.

## Running

Run these commands in order to execute the pipeline:

```bash
uv sync
uv run download_datasets.py
uv run run_pipeline.py
```

`uv sync` creates a local `.venv` and installs everything from
`pyproject.toml`. Each `uv run` step then runs in that environment
without needing it activated.

## Results

The results here are for demonstration purposes only; they act as a
public proxy for the non-disclosure production fabrication data.

`run_pipeline.py` writes per-dataset RMSE for both models to
`demo_results.csv`, and `cart.py` saves each dataset's fitted tree to
`figures/<dataset>/cart_tree.png`.

| Dataset | Leaves | Global RMSE | Data-Driven RMSE | Improvement |
|---|---|---|---|---|
| naval_propulsion_plant | 10 | 0.000544 | 0.000275 | +49.34% |
| video_transcoding | 14 | 0.838 | 0.794 | +5.20% |
| auction_verification | 2 | 487.60 | 444.16 | +8.91% |
| socmob | 2 | 14.92 | 13.10 | +12.22% |
| airfoil_self_noise | 2 | 1.416 | 1.299 | +8.32% |

CART settings: the minimum number of samples per leaf is 5% of the
training set, bounded between 300 and 800, and the number of leaves is
selected from 2 to 15 via 3-fold cross-validation with the 1-SE rule
(see `cart_auto_tuning.py`).

In the paper, partitioning improves performance on heterogeneous
fabrication data. These public datasets illustrate the same effect on
data from other fields.

## Data sources

- `naval_propulsion_plant`: UCI Machine Learning Repository, DOI 10.24432/C5K31K (CC BY 4.0)
- `video_transcoding`: UCI Machine Learning Repository, DOI 10.24432/C58C9K (CC BY 4.0); OpenML version, ID 44974
- `auction_verification`: UCI Machine Learning Repository, DOI 10.24432/C52K6N (CC BY 4.0)
- `socmob`: OpenML ID 44987, originally from StatLib; Biblarz and Raftery (1993), DOI 10.2307/2096220. Non-commercial scholarly and teaching use only.
- `airfoil_self_noise`: UCI Machine Learning Repository, DOI 10.24432/C5VW2C (CC BY 4.0)

*Note: AutoGluon training is not fully deterministic, so results may
differ slightly on other hardware or software versions.*
