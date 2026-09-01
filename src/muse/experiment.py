"""Model-update alert-rate-stability experiment.

Trains an "old" and a "new" model once, on the same training window, with
different hyperparameters -- a model replacement via capacity/tuning, not
via training recency. The two models are frozen and never retrained. Every
later month becomes one walk-forward **evaluation step**, at which each
predictor variant's quantile mapping is refit (or not) per its own policy,
and every predictor is scored on that step's month. Because the models
never change across the sweep, this isolates mapping staleness/refit-
cadence from model staleness.

Predictor variants:

  - p1       old model, mapping fit once right after training, never refit.
  - p1_5     new model, but paired with p1's stale mapping.
  - p2       new model, mapping refit each step on the single nearest
             prior month.
  - custom   new model, mapping refit each step on an expanding window
             starting at the last training month (always >=2 months wide,
             so it never numerically coincides with p2, even at step 1).
  - default  new model, but with its own dedicated "fit once, never
             revisited" mapping, fit on a third month distinct from
             p1/p1_5's and p2/custom's -- "we set up a mapping for the new
             model at launch and never refreshed it", as opposed to
             p1_5's "we swapped models but kept the OLD model's mapping".
  - raw      new model's scores with no mapping applied at all.

The metric is **alert-rate relative error**: for each of the 4 interior
edges in BIN_EDGES (0.2, 0.5, 0.6, 0.8), `(observed_alert_rate -
target_alert_rate) / target_alert_rate * 100`. This is the quantity a
downstream fixed-threshold alerting rule actually cares about: does a
fixed threshold still fire at (approximately) the rate it's supposed to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from muse.models import TrainedModel, train_model
from muse.quantile_mapping import (
    BIN_EDGES,
    DEFAULT_TARGET_ALERT_RATES,
    QuantileMapping,
)

# Interior cumulative edges (0.0 excluded -- always 1.0 alert rate by
# definition -- 1.0 excluded -- no threshold above the top edge). These
# are the actual fixed-threshold operating points a downstream alerting
# rule would use (e.g. "flag everything scoring >= the 0.6 threshold"),
# so alert-rate error is measured at each of these, not per BIN.
INTERIOR_EDGES = [e for e in BIN_EDGES if e not in (0.0, 1.0)]
N_EDGES = len(INTERIOR_EDGES)
EDGE_LABELS = [f">= {e:.1f}" for e in INTERIOR_EDGES]

MIN_EVENTS_PER_FIT = 200
MIN_EVENTS_PER_EVAL = 200


def model_score_correlation(old_model: TrainedModel, new_model: TrainedModel, scoring_df: pd.DataFrame) -> float:
    """Pearson correlation between old_model's and new_model's corrected
    scores on the SAME rows (`scoring_df`).

    This is the "checkable" number referenced by run_experiment.py/the
    README when justifying how wide the old/new hyperparameter gap needs
    to be: a mild gap (e.g. n_estimators alone, on identical training
    data) leaves LightGBM's scores highly correlated even though the
    hyperparameters differ, which understates p1_5's staleness effect
    (see run_walk_forward_experiment's docstring). Low correlation here
    means the two models genuinely disagree about how to rank rows, which
    is what makes a mapping fit for one model a poor fit for the other.
    """
    old_scores = old_model.corrected_scores(scoring_df)
    new_scores = new_model.corrected_scores(scoring_df)
    return float(np.corrcoef(old_scores, new_scores)[0, 1])


def target_alert_rate_curve(target_alert_rates: dict[float, float] | None = None) -> np.ndarray:
    """Target cumulative alert rate at each of INTERIOR_EDGES, in order."""
    if target_alert_rates is None:
        target_alert_rates = DEFAULT_TARGET_ALERT_RATES
    return np.array([target_alert_rates[e] for e in INTERIOR_EDGES])


@dataclass
class PredictorResult:
    name: str
    scores: np.ndarray  # raw model scores on the evaluation window
    mapping: QuantileMapping  # mapping applied to get bucket assignments
    n_events: int

    def alert_rates(self) -> np.ndarray:
        """Observed alert rate at each of INTERIOR_EDGES, in order."""
        return np.array([self.mapping.alert_rate(self.scores, e) for e in INTERIOR_EDGES])

    def alert_rate(self, edge: float) -> float:
        return self.mapping.alert_rate(self.scores, edge)


def relative_error_table(predictors: dict[str, PredictorResult]) -> pd.DataFrame:
    """One row per (predictor, edge): observed alert rate, target alert
    rate, and the alert-rate relative error, for every predictor variant.
    This is the "does a fixed threshold at this edge still fire at the
    rate it's supposed to" question -- the actual quantity a downstream
    fixed-threshold alerting rule cares about, as opposed to per-bin
    population density (which can look wrong in a bin even while the
    cumulative alert rate at the edges either side of it stays on target).
    """
    target_curve = target_alert_rate_curve()
    rows = []
    for name, result in predictors.items():
        observed_curve = result.alert_rates()
        for e_idx, edge in enumerate(INTERIOR_EDGES):
            target = target_curve[e_idx]
            observed = observed_curve[e_idx]
            rel_err = (observed - target) / target * 100.0 if target > 1e-10 else (observed - target) * 1000.0
            rows.append(
                {
                    "predictor": name,
                    "bucket_id": e_idx,
                    "bin_label": EDGE_LABELS[e_idx],
                    "edge": edge,
                    "n_events": result.n_events,
                    "alert_rate": observed,
                    "target_alert_rate": target,
                    "relative_error_pct": rel_err,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_relative_error_table(
    predictors: dict[str, PredictorResult], n_bootstrap: int = 500, seed: int = 0
) -> pd.DataFrame:
    """Bootstrap-resampled alert-rate relative error, giving each
    (predictor, edge) a distribution rather than a single point estimate.
    """
    rng = np.random.default_rng(seed)
    target_curve = target_alert_rate_curve()
    n = next(iter(predictors.values())).n_events
    rows = []
    for b_iter in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        for name, result in predictors.items():
            resampled_scores = result.scores[idx]
            for e_idx, edge in enumerate(INTERIOR_EDGES):
                target = target_curve[e_idx]
                observed = result.mapping.alert_rate(resampled_scores, edge)
                rel_err = (observed - target) / target * 100.0 if target > 1e-10 else (observed - target) * 1000.0
                rows.append(
                    {
                        "predictor": name,
                        "bucket_id": e_idx,
                        "bin_label": EDGE_LABELS[e_idx],
                        "relative_error_pct": rel_err,
                        "bootstrap_iter": b_iter,
                    }
                )
    return pd.DataFrame(rows)


def fixed_threshold_table(predictors: dict[str, PredictorResult], edge: float = 0.6) -> pd.DataFrame:
    """Alert rate at a fixed cumulative bin edge, for each predictor."""
    rows = []
    for name, result in predictors.items():
        rate = result.alert_rate(edge)
        rows.append(
            {
                "predictor": name,
                "edge": edge,
                "alert_rate": rate,
                "n_alerts": int(round(rate * result.n_events)),
                "n_events": result.n_events,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ModelUpdateExperiment:
    """Holds the two (already-trained, frozen) models, the quantile
    mappings refit for this evaluation step, and the evaluation results."""

    fit_window_label: str
    eval_window_label: str
    old_model: TrainedModel  # p1's model
    new_model: TrainedModel  # p1_5/p2/custom/raw/default's model
    mapping_old: QuantileMapping  # fit on old_model's fit-window scores (p1/p1_5)
    mapping_new: QuantileMapping  # fit on new_model's fit-window scores (p2)
    predictors: dict[str, PredictorResult] = field(default_factory=dict)

    def relative_error_table(self) -> pd.DataFrame:
        return relative_error_table(self.predictors)

    def bootstrap_relative_error_table(self, n_bootstrap: int = 500, seed: int = 0) -> pd.DataFrame:
        return bootstrap_relative_error_table(self.predictors, n_bootstrap=n_bootstrap, seed=seed)

    def fixed_threshold_table(self, edge: float = 0.6) -> pd.DataFrame:
        return fixed_threshold_table(self.predictors, edge=edge)

    def drift_baseline_table(self, mapping_fit_df: pd.DataFrame) -> pd.DataFrame:
        """How much of p1's own alert-rate accuracy changes between the
        month its mapping was fit on and the eval month, with **no model
        swap at all**? Re-scores `mapping_fit_df` itself through p1's exact
        recipe (old_model + mapping_old) and compares its in-window
        relative error to its held-out relative error, isolating ordinary
        temporal drift from the model-swap effect measured by
        relative_error_table().
        """
        fit_window_scores = self.old_model.corrected_scores(mapping_fit_df)
        fit_window_rates = np.array([self.mapping_old.alert_rate(fit_window_scores, e) for e in INTERIOR_EDGES])

        eval_rates = self.predictors["p1"].alert_rates()
        target_curve = target_alert_rate_curve()

        rows = []
        for e_idx, edge in enumerate(INTERIOR_EDGES):
            target = target_curve[e_idx]
            fit_err = (
                (fit_window_rates[e_idx] - target) / target * 100.0
                if target > 1e-10
                else (fit_window_rates[e_idx] - target) * 1000.0
            )
            eval_err = (
                (eval_rates[e_idx] - target) / target * 100.0
                if target > 1e-10
                else (eval_rates[e_idx] - target) * 1000.0
            )
            rows.append(
                {
                    "bucket_id": e_idx,
                    "bin_label": EDGE_LABELS[e_idx],
                    "edge": edge,
                    "target_alert_rate": target,
                    "fit_window_alert_rate": fit_window_rates[e_idx],
                    "eval_window_alert_rate": eval_rates[e_idx],
                    "fit_window_relative_error_pct": fit_err,
                    "temporal_drift_relative_error_pct": eval_err,
                }
            )
        return pd.DataFrame(rows)


def run_model_update_experiment(
    old_model: TrainedModel,
    new_model: TrainedModel,
    mapping_new_fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    fit_window_label: str,
    eval_window_label: str,
    mapping_old: QuantileMapping,
    mapping_default: QuantileMapping,
    target_alert_rates: dict[float, float] | None = None,
    mapping_custom_fit_df: pd.DataFrame | None = None,
) -> ModelUpdateExperiment:
    """Evaluate one walk-forward step for two already-trained, frozen
    models (see module docstring for the six predictor variants).

    `mapping_old` (p1/p1_5) and `mapping_default` (default) are fixed,
    already-fit mappings passed in unchanged. `mapping_new` (p2) is
    refit here on `mapping_new_fit_df` alone -- the single nearest prior
    month. `mapping_custom` (custom) is refit here on
    `mapping_custom_fit_df`, a wider window; if omitted, it defaults to
    `mapping_new_fit_df` (so custom == p2 for callers that don't need the
    distinction).
    """
    mapping_new = QuantileMapping.fit(new_model.corrected_scores(mapping_new_fit_df), target_alert_rates)
    raw_mapping = QuantileMapping(np.array(BIN_EDGES[1:-1]), mapping_new.target_alert_rates)

    if mapping_custom_fit_df is None:
        mapping_custom = mapping_new
    else:
        mapping_custom = QuantileMapping.fit(new_model.corrected_scores(mapping_custom_fit_df), target_alert_rates)

    eval_scores_old = old_model.corrected_scores(eval_df)
    eval_scores_new = new_model.corrected_scores(eval_df)

    predictors = {
        "p1": PredictorResult("p1", eval_scores_old, mapping_old, len(eval_df)),
        "p1_5": PredictorResult("p1_5", eval_scores_new, mapping_old, len(eval_df)),
        "p2": PredictorResult("p2", eval_scores_new, mapping_new, len(eval_df)),
        # custom's window is always wider than p2's single month, so its
        # numbers are never identical to p2's.
        "custom": PredictorResult("custom", eval_scores_new, mapping_custom, len(eval_df)),
        "default": PredictorResult("default", eval_scores_new, mapping_default, len(eval_df)),
        "raw": PredictorResult("raw", eval_scores_new, raw_mapping, len(eval_df)),
    }

    return ModelUpdateExperiment(
        fit_window_label=fit_window_label,
        eval_window_label=eval_window_label,
        old_model=old_model,
        new_model=new_model,
        mapping_old=mapping_old,
        mapping_new=mapping_new,
        predictors=predictors,
    )


def run_walk_forward_experiment(
    df: pd.DataFrame,
    month_column: str = "month",
    model_train_months: list[int] | None = None,
    eval_months: list[int] | None = None,
    beta_old: float = 0.10,
    beta_new: float = 0.10,
    n_estimators_old: int = 200,
    n_estimators_new: int = 200,
    max_depth_old: int = -1,
    max_depth_new: int = -1,
    num_leaves_old: int = 31,
    num_leaves_new: int = 31,
    learning_rate_old: float = 0.1,
    learning_rate_new: float = 0.1,
    seed: int = 0,
    target_alert_rates: dict[float, float] | None = None,
    min_events_per_fit: int = MIN_EVENTS_PER_FIT,
    min_events_per_eval: int = MIN_EVENTS_PER_EVAL,
) -> pd.DataFrame:
    """Train old/new models once on `model_train_months`, then walk
    forward over every later month as one evaluation step, refitting each
    predictor's mapping per its own policy (see module docstring for the
    six predictor variants and their refit policies).

    The hyperparameter gap between old/new must be wide enough that their
    scores actually diverge -- a mild gap (e.g. n_estimators alone) tends
    to leave LightGBM's scores highly correlated (~0.8) even on identical
    training data, which understates p1_5's staleness effect. See
    run_experiment.py's defaults for a gap wide enough to matter, and
    `model_score_correlation`/scripts/check_model_divergence.py to check
    the resulting correlation directly.

    `eval_months` must all fall strictly after `model_train_months` and
    the mapping-fit month (the month right after training, used to fit
    both mapping_old and default's mapping) -- no evaluation step may
    score a model on data it or its initial mapping has already seen.
    `custom`'s per-step mapping is refit on an expanding window starting
    at the LAST training month (reused only to fit a mapping, never to
    train a model, so this does not violate that guard) so it is always
    at least 2 months wide, even at the first evaluation step.

    By default (adapted to BAF's 8 months): both models train on months
    0-1, the initial mappings are fit on month 2, and months 3-7 are each
    one walk-forward evaluation step.

    Returns a long DataFrame (one row per predictor x edge x eval step)
    with `point_idx`/`mapping_new_fit_month`/`eval_month` columns added.
    """
    months = sorted(int(m) for m in df[month_column].unique())
    if model_train_months is None:
        model_train_months = months[:2]

    mapping_old_fit_month = max(model_train_months) + 1
    if mapping_old_fit_month not in months:
        raise ValueError(
            f"month {mapping_old_fit_month} (immediately after model_train_months) is not present in "
            f"the data; cannot fit mapping_old/the first mapping_new. Available months: {months}"
        )

    if eval_months is None:
        eval_months = [m for m in months if m > mapping_old_fit_month]
    else:
        overlapping = [m for m in eval_months if m in model_train_months or m == mapping_old_fit_month]
        if overlapping:
            raise ValueError(
                f"eval_months {overlapping} overlap model_train_months/the mapping-fit month "
                f"({mapping_old_fit_month}) -- evaluation must start strictly after both models' "
                f"training window and the initial mapping fit, not inside or on it."
            )
    if not eval_months:
        raise ValueError(f"no months left to evaluate on after the model-training/mapping-fit months: {months}")

    train_df = df[df[month_column].isin(model_train_months)].reset_index(drop=True)
    if len(train_df) < min_events_per_fit:
        raise ValueError("not enough rows in model_train_months")

    old_model = train_model(
        train_df, beta=beta_old, seed=seed, n_estimators=n_estimators_old,
        max_depth=max_depth_old, num_leaves=num_leaves_old, learning_rate=learning_rate_old,
    )
    new_model = train_model(
        train_df, beta=beta_new, seed=seed, n_estimators=n_estimators_new,
        max_depth=max_depth_new, num_leaves=num_leaves_new, learning_rate=learning_rate_new,
    )

    # mapping_old is fit on the same single month, immediately after the
    # shared training window, that new_model's mapping would also use as
    # its first per-step fit month -- both models' initial mapping fit is
    # on identical footing (same data, same recency relative to training).
    mapping_old_fit_df = df[df[month_column] == mapping_old_fit_month].reset_index(drop=True)
    if len(mapping_old_fit_df) < min_events_per_fit:
        raise ValueError(f"not enough rows in month {mapping_old_fit_month} to fit mapping_old")
    mapping_old = QuantileMapping.fit(old_model.corrected_scores(mapping_old_fit_df), target_alert_rates)

    # default's mapping is fit once, on a third month distinct from
    # mapping_old's and every per-step mapping_new/mapping_custom month,
    # so it never numerically coincides with any other predictor.
    mapping_default_fit_month = min(model_train_months)
    mapping_default_fit_df = df[df[month_column] == mapping_default_fit_month].reset_index(drop=True)
    if len(mapping_default_fit_df) < min_events_per_fit:
        raise ValueError(f"not enough rows in month {mapping_default_fit_month} to fit mapping_default")
    mapping_default = QuantileMapping.fit(new_model.corrected_scores(mapping_default_fit_df), target_alert_rates)

    rows = []
    # Starts one month before mapping_old_fit_month (reused only to fit a
    # mapping, not to train a model) so custom's window is always >=2
    # months wide -- otherwise it would collapse to a single month,
    # identical to p2's, at the very first evaluation step.
    mapping_custom_fit_start_month = max(model_train_months)
    for point_idx, eval_month in enumerate(eval_months):
        mapping_new_fit_month = eval_month - 1
        mapping_new_fit_df = df[df[month_column] == mapping_new_fit_month].reset_index(drop=True)
        mapping_custom_fit_months = [
            m for m in months if mapping_custom_fit_start_month <= m <= mapping_new_fit_month
        ]
        mapping_custom_fit_df = df[df[month_column].isin(mapping_custom_fit_months)].reset_index(drop=True)
        eval_df = df[df[month_column] == eval_month].reset_index(drop=True)
        if len(mapping_new_fit_df) < min_events_per_fit or len(eval_df) < min_events_per_eval:
            continue

        exp = run_model_update_experiment(
            old_model=old_model,
            new_model=new_model,
            mapping_new_fit_df=mapping_new_fit_df,
            eval_df=eval_df,
            fit_window_label=f"month {mapping_new_fit_month}",
            eval_window_label=f"month {eval_month}",
            mapping_old=mapping_old,
            mapping_default=mapping_default,
            target_alert_rates=target_alert_rates,
            mapping_custom_fit_df=mapping_custom_fit_df,
        )

        point_df = exp.relative_error_table()
        point_df["point_idx"] = point_idx
        point_df["mapping_new_fit_month"] = mapping_new_fit_month
        point_df["eval_month"] = eval_month
        rows.append(point_df)

    if not rows:
        raise ValueError("no eval step had enough fit/eval events -- check min_events thresholds")
    return pd.concat(rows, ignore_index=True)
