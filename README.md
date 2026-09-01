# Quantile Mapping — Model Update Stability Study

A self-contained study and reference implementation of **quantile
mapping** for keeping fixed-threshold alerting rules stable across model
updates, demonstrated end to end on the public [Bank Account Fraud
(BAF)](https://github.com/feedzai/bank-account-fraud) dataset.

## The problem

Downstream systems often convert a model's score into an action using a
fixed, manually chosen threshold (e.g. "flag anything scoring above 0.7").
These thresholds, in fraud detection, are chosen to mantain a desired
alert rate (percentage of flagged events), based on analyst capacity.
Whenever the model behind that score is retrained or replaced, its raw
score distribution usually shifts, even if predictive quality stays the
same or improves — so a threshold tuned for the old model can silently
start flagging a very different fraction of events under the new one.
Re-tuning every downstream threshold by hand every time a model changes
does not scale, especially when many independent consumers each rely on
their own threshold.

## The idea

Fit a **quantile mapping**: a set of score thresholds, one per fixed
alert-rate edge, chosen so that a model's raw scores clear each threshold
at a fixed, pre-specified target rate. Because the mapping is refit per
model version (using only the new model's own scores, no labels
required), the fraction of events above any given threshold — the alert
rate — stays approximately constant across model updates, without any
downstream threshold ever needing to change.

```
mapped_score = quantile_mapping(posterior_correction(model(x)))
```

`posterior_correction` is an optional first step that corrects for score
inflation caused by training on an undersampled majority class — a common
practice in imbalanced classification (see `posterior_correction.py`).

## What this study demonstrates

An "old" and a "new" model are trained **once**, up front, on the **same**
training window, with a **wide hyperparameter gap** — a shallow,
under-tuned "legacy" model vs. a properly tuned, higher-capacity one
(different `max_depth`/`num_leaves`/`learning_rate`/`n_estimators`/
undersampling ratio). The gap needs to be wide: a mild difference
(e.g. `n_estimators` alone) leaves the two models' raw scores highly
correlated (~0.8, measured on this dataset) even on identical training
data, understating how badly a mapping calibrated for one model fits the
other. Using the *same* training window for both also means neither model
has a home-turf recency advantage over the other when their initial
mappings are fit. The two models are then frozen and never retrained
again. `p1`/`p1_5`'s quantile mapping is **also** fit once, on the single
month right after training, and never revisited — a production mapping
that stops being maintained once the old model is retired.

Every later month — strictly after the training window and that initial
mapping-fit month, so no evaluation step scores a model on data it or its
mapping has already seen — becomes one walk-forward **evaluation step**:
`p2`'s mapping is refit fresh at each step on just the month immediately
before it; `custom`'s mapping is refit fresh at each step on an expanding
window starting at the last training month; `p1`'s model+mapping and
`p1_5`'s mapping never change again.

Six predictor variants are evaluated at every step, and none of them
numerically coincide with any other, at any evaluation step including
the first one (each uses either a different model, a different mapping,
or a mapping fit on a different month than every other predictor's):

| Predictor | Model | Quantile mapping |
|---|---|---|
| `p1` | an "old" model, trained once | `mapping_old`, fit once on the month right after training, never refit |
| `p1_5` | the "new" model, trained once on the SAME window with different hyperparameters | **not refit** — paired with `p1`'s `mapping_old`, fit for a different model and never revisited: the worst-case combination |
| `p2` | the same "new" model | `mapping_new`, **refit every step**, on the nearest prior month's data |
| `custom` | new model | `mapping_custom`, **refit every step**, on an expanding window starting at the last training month (always ≥2 months wide) |
| `default` | new model | `mapping_default`, its own dedicated mapping fit once (on a month distinct from `p1`/`p1_5`'s, `p2`'s, and `custom`'s), never revisited |
| `raw` | new model | no mapping at all |

**Result** (3 BAF variants — `Base`, `Variant II`, `Variant IV` — each
run with 3 seeds (0, 1, 2); both models trained on months 0–1 — old:
`max_depth=1, num_leaves=2, learning_rate=0.5, n_estimators=20,
beta=0.05`; new: `max_depth=10, num_leaves=128, learning_rate=0.03,
n_estimators=500, beta=0.50` — `mapping_old`/the first `mapping_new` fit
on month 2, 5 walk-forward evaluation steps across months 3–7 per run):
median absolute **alert-rate** relative error — i.e.
`(observed_alert_rate - target_alert_rate) / target_alert_rate * 100` at
each of the 4 fixed threshold edges (0.2, 0.5, 0.6, 0.8) — across all 4
edges, every evaluation step, and every seed within a variant (median /
range across the 3 seeds):

| Predictor | `Base` | `Variant II` | `Variant IV` |
|---|---|---|---|
| `p2` | **14.4%** [14.4, 15.0] | **8.7%** [8.6, 9.7] | **9.5%** [9.3, 10.7] |
| `custom` | **15.0%** [13.9, 15.2] | **7.0%** [6.6, 7.2] | **7.9%** [7.0, 8.3] |
| `default` | 16.6% [15.2, 16.9] | 7.6% [6.4, 9.1] | 7.5% [6.0, 8.0] |
| `p1` | 16.5% [15.0, 18.1] | 18.8% [18.1, 19.1] | 16.3% [13.6, 16.7] |
| `p1_5` | 47.0% [45.5, 47.1] | 39.5% [37.9, 40.9] | 55.5% [52.8, 56.1] |
| `raw` | 99.7% [99.7, 99.8] | 99.8% [99.8, 99.8] | 99.8% [99.8, 99.8] |

`p2` and `custom` — the two predictors whose mapping is refit every
step — are consistently the best across all 3 variants and all 3 seeds,
with tight seed-to-seed spread. `p1_5` (new model scored through `p1`'s
stale, never-refit mapping) is consistently the worst *calibrated*
predictor, at roughly 2-7x every other calibrated predictor's error
(closest to `p1` — the other never-refit mapping — and furthest from
`p2`/`custom`), but `raw` (no mapping at all) is far worse still, at
~100% — roughly 1.8-2.5x `p1_5`'s error and 7-14x `p2`/`custom`'s. This
is not a single-seed or single-variant artifact: see
`results/summary_all_variants.json` for the full per-seed breakdown, and
`scripts/run_experiment.py --variant ... --seeds ...` to reproduce or
extend the sweep to more BAF variants/seeds.

Alert-rate error (measured at the 4 actual operating thresholds a
downstream fixed-threshold rule would use) is deliberately a more
conservative metric than raw per-bin population density: errors in
adjacent bins can partially offset when you look at the *cumulative*
alert rate at a threshold rather than each bin's density in isolation, so
this number tends to be smaller (and more representative of real
operational impact) than a bin-density-based metric would show for the
same predictors. The per-edge breakdown for `Base` makes the pattern
clear (median relative error per edge, %):

| Predictor | `>= 0.2` | `>= 0.5` | `>= 0.6` | `>= 0.8` |
|---|---|---|---|---|
| `p1` | 0.2 | 11.5 | 42.3 | 54.6 |
| `p1_5` | -63.5 | -56.7 | -25.3 | -0.1 |
| `p2` | -0.3 | 2.2 | 4.2 | 2.3 |
| `custom` | -0.5 | -1.1 | 0.3 | -0.8 |
| `default` | -0.7 | 4.4 | 16.6 | 23.4 |
| `raw` | -99.2 | -99.8 | -99.7 | -99.9 |

`p2` and `custom` (mappings refit every step) stay within a few percent
of target at every edge. `p1` and `default` (mappings fit once, never
refit) drift increasingly far from target as the edge moves up the score
range — expected, since higher edges are more sensitive to distributional
shift in the tail. `p1_5` (new model scored through `p1`'s mapping, fit
for a very different, much shallower model) is badly wrong at the low
edges (-64%, -57%) and only converges toward target at the highest edge —
exactly what "pairing a model with a mapping calibrated for a very
different model" should look like: not a uniform shift, but a
threshold-dependent distortion. `raw` (no mapping at all) collapses to
roughly -100% at every edge, since the new model's raw probability output
is nowhere near the target's alert-rate scale — a stark illustration of
why a fixed threshold on an uncalibrated score is meaningless on its own.
See `notebooks/results.ipynb` for a single-variant, single-seed
walkthrough with narrative and figures, and `results/<variant>/` (after
running `scripts/run_experiment.py`) for the full sweep's plots (a
model-update box plot, a calibration-only box plot, and a fixed-threshold
alert-rate bar chart, per variant).

## Repository layout

```
src/muse/
    quantile_mapping.py       percentile-fit quantile mapping against a target alert-rate curve
    posterior_correction.py   undersampling-bias correction
    data.py                   BAF loading (via kagglehub)
    models.py                 LightGBM training with majority-class undersampling
    experiment.py             walk-forward driver + p1/p1_5/p2/custom/default/raw orchestration
    plotting.py               colorblind-safe, hatched, symlog figures
scripts/
    download_data.py            one-time BAF download
    run_experiment.py           end-to-end CLI run across variants/seeds, writes results/<variant>/ + figures
    check_model_divergence.py   reproduces the old/new score correlation numbers quoted in this README
notebooks/
    results.ipynb             a single-variant, single-seed walkthrough, with narrative and plots
tests/
    test_quantile_mapping.py  unit tests for the transform itself
    test_experiment.py        correctness properties on synthetic data
```

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest tests/ -v                      # fast, no data download needed

uv run python scripts/download_data.py       # downloads BAF via kagglehub (~530MB)
uv run python scripts/run_experiment.py --variant Base --seeds 0
uv run python scripts/run_experiment.py --variant Base "Variant II" "Variant IV" --seeds 0 1 2
uv run python scripts/check_model_divergence.py   # reproduce the old/new score correlation numbers
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
```

`download_data.py` uses `kagglehub`, which can download the dataset even
without an explicit Kaggle API token in most environments; if it fails,
create an API token at kaggle.com → Settings → "Create New Token" and save
it to `~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME`/`KAGGLE_KEY`.

---

## About the dataset

This study runs on the **Bank Account Fraud (BAF)** dataset suite.

**The public dataset suite is available for download through
[Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).**

> The paper describing this dataset suite, *"Turning the Tables: Biased,
> Imbalanced, Dynamic Tabular Datasets for ML Evaluation,"* was accepted at
> **NeurIPS 2022**.
> [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html) ·
> [Datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf) ·
> [Source repository](https://github.com/feedzai/bank-account-fraud)
