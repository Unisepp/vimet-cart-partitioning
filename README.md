# Partitioning Strategies for Improving AutoML Performance on Heterogeneous Virtual Metrology Data: Public Demo (CTR-23)

Reproduces the paper's public demonstration: Global vs. Data-Driven
RMSE on 5 datasets from the OpenML CTR-23 regression
benchmark suite (suite 353):

```
naval_propulsion_plant
video_transcoding
auction_verification
socmob
airfoil_self_noise
```

## Approach

For each dataset, two models are trained on the same 80/20 train/test
split and compared on held-out RMSE:

- **Global**: one AutoGluon predictor trained on the full training set.
- **Data-Driven**: a `DecisionTreeRegressor` partitions
  the training data into leaves, and a separate AutoGluon predictor is
  trained per leaf.


## Files

- `download_demo_datasets.py` - downloads the 5 datasets above from
  suite 353 into `data/<dataset>/raw.csv`.
- `data_generic.py` — train/test split (80/20, seed 42).
- `cart_auto_tuning.py` — hyperparameter selection for CART
- `cart_generic.py` — fits the CART tree and assigns each row a
  `leaf_id`.
- `train_global_generic.py` — trains the Global AutoGluon model.
- `train_cart_leaf_generic.py` — trains one AutoGluon model per leaf.
- `evaluate_generic.py` — scores both models on the test set; writes
  per-row and per-leaf RMSE breakdowns.
- `run_demo_pipeline.py` — runs the full pipeline (split → CART →
  train → evaluate) across all 5 datasets and writes
  `demo_results.csv`.

## Running

```bash
pip install -r requirements.txt
python download_demo_datasets.py
python run_demo_pipeline.py
```


