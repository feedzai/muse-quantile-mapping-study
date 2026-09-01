"""Self-checks for the model-update experiment orchestration.

Uses small synthetic data (not real BAF) so the test suite runs in under a
second and does not depend on network access / Kaggle credentials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from muse.experiment import (
    model_score_correlation,
    run_model_update_experiment,
    run_walk_forward_experiment,
    target_alert_rate_curve,
)
from muse.models import train_model, undersample_majority
from muse.quantile_mapping import QuantileMapping


def _synthetic_baf_like(
    n: int, seed: int, fraud_rate: float = 0.02, month: int = 0, signal_scale: float = 1.0
) -> pd.DataFrame:
    """A small DataFrame with BAF's column names/dtypes so train_model()'s
    feature/encoder logic exercises the real code path without downloading
    the actual dataset.

    `signal_scale` widens the separation between the fraud/legit signal
    distributions (default 1.0 = original spread); used to simulate a later
    window where the model's raw score distribution has genuinely shifted,
    so a stale vs. freshly-refit quantile mapping can actually be told
    apart.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    fraud_bool = np.concatenate([np.ones(n_fraud), np.zeros(n - n_fraud)]).astype(int)
    rng.shuffle(fraud_bool)

    # A couple of numeric features correlated with the label, so the model
    # actually learns something (rather than pure noise) -- needed for the
    # quantile-mapping comparison below to be meaningful.
    signal = fraud_bool * rng.normal(3 * signal_scale, 1, n) + (1 - fraud_bool) * rng.normal(0, 1, n)

    return pd.DataFrame(
        {
            "fraud_bool": fraud_bool,
            "income": signal + rng.normal(0, 0.5, n),
            "name_email_similarity": rng.uniform(0, 1, n),
            "prev_address_months_count": rng.integers(0, 100, n),
            "current_address_months_count": rng.integers(0, 100, n),
            "customer_age": rng.integers(18, 90, n),
            "days_since_request": rng.exponential(1, n),
            "intended_balcon_amount": rng.normal(0, 10, n),
            "payment_type": rng.choice(["AA", "AB", "AC"], n),
            "zip_count_4w": rng.integers(1, 500, n),
            "velocity_6h": signal + rng.normal(0, 1, n),
            "velocity_24h": rng.normal(0, 1, n),
            "velocity_4w": rng.normal(0, 1, n),
            "bank_branch_count_8w": rng.integers(0, 50, n),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 10, n),
            "employment_status": rng.choice(["CA", "CB", "CC"], n),
            "credit_risk_score": rng.integers(-100, 400, n),
            "email_is_free": rng.integers(0, 2, n),
            "housing_status": rng.choice(["BA", "BB"], n),
            "phone_home_valid": rng.integers(0, 2, n),
            "phone_mobile_valid": rng.integers(0, 2, n),
            "bank_months_count": rng.integers(-1, 30, n),
            "has_other_cards": rng.integers(0, 2, n),
            "proposed_credit_limit": rng.choice([200, 500, 1000], n).astype(float),
            "foreign_request": rng.integers(0, 2, n),
            "source": rng.choice(["INTERNET", "TELEAPP"], n),
            "session_length_in_minutes": rng.exponential(5, n),
            "device_os": rng.choice(["windows", "linux", "macintosh"], n),
            "keep_alive_session": rng.integers(0, 2, n),
            "device_distinct_emails_8w": rng.integers(0, 3, n),
            "device_fraud_count": np.zeros(n, dtype=int),
            "month": month,
        }
    )


def test_undersample_majority_keeps_all_fraud_and_reduces_legit():
    df = _synthetic_baf_like(2000, seed=0, fraud_rate=0.1)
    out = undersample_majority(df, beta=0.2, seed=1)
    n_fraud = (df["fraud_bool"] == 1).sum()
    assert (out["fraud_bool"] == 1).sum() == n_fraud
    assert (out["fraud_bool"] == 0).sum() < (df["fraud_bool"] == 0).sum()


def test_undersample_majority_beta_one_is_noop():
    df = _synthetic_baf_like(500, seed=0)
    out = undersample_majority(df, beta=1.0, seed=1)
    assert len(out) == len(df)


def test_model_score_correlation_is_one_for_the_same_model():
    """model_score_correlation must be exactly 1.0 when old_model and
    new_model are literally the same trained model (sanity check on the
    correlation computation itself, independent of any experiment code)."""
    train_df = _synthetic_baf_like(1500, seed=0, fraud_rate=0.05)
    scoring_df = _synthetic_baf_like(500, seed=1, fraud_rate=0.05)
    model = train_model(train_df, beta=0.5, seed=0, n_estimators=20)
    corr = model_score_correlation(model, model, scoring_df)
    assert corr == pytest.approx(1.0, abs=1e-9)


def test_model_score_correlation_drops_with_a_wider_hyperparameter_gap():
    """A wider hyperparameter gap between old/new must not INCREASE their
    score correlation on the same data -- this is the property
    run_experiment.py's hyperparameter choice relies on (see
    scripts/check_model_divergence.py and the README's sensitivity note)."""
    train_df = _synthetic_baf_like(1500, seed=0, fraud_rate=0.05)
    scoring_df = _synthetic_baf_like(500, seed=1, fraud_rate=0.05)

    mild_old = train_model(train_df, beta=0.10, seed=0, n_estimators=20)
    mild_new = train_model(train_df, beta=0.40, seed=0, n_estimators=40)
    mild_corr = model_score_correlation(mild_old, mild_new, scoring_df)

    wide_old = train_model(train_df, beta=0.05, seed=0, n_estimators=5, max_depth=1, num_leaves=2)
    wide_new = train_model(train_df, beta=0.80, seed=0, n_estimators=80, max_depth=8, num_leaves=64)
    wide_corr = model_score_correlation(wide_old, wide_new, scoring_df)

    assert wide_corr <= mild_corr


def test_target_alert_rate_curve_matches_default_target_alert_rates():
    """target_alert_rate_curve()[i] must equal DEFAULT_TARGET_ALERT_RATES at
    the corresponding interior edge, in the same order as INTERIOR_EDGES
    (sanity check that it is not silently using some other curve)."""
    from muse.experiment import INTERIOR_EDGES
    from muse.quantile_mapping import DEFAULT_TARGET_ALERT_RATES

    curve = target_alert_rate_curve()
    assert len(curve) == len(INTERIOR_EDGES)
    for i, edge in enumerate(INTERIOR_EDGES):
        assert curve[i] == pytest.approx(DEFAULT_TARGET_ALERT_RATES[edge], abs=1e-9)
    # The curve must be non-increasing (higher thresholds alert less often).
    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1))


@pytest.mark.slow
def test_p2_tracks_target_better_than_p1_5():
    """Core correctness property this study exists to demonstrate: refitting
    the quantile mapping (p2) should track the target bin density much more
    closely than keeping the stale mapping (p1_5) does. The old and new
    models here share the same architecture and hyperparameters and are
    both trained once, up front, on overlapping early windows -- a routine
    retrain-with-more-data update, which is a simpler/different setup than
    the same-training-window, different-hyperparameters design used for
    the real BAF run, but sufficient here since only the quantile MAPPING
    refit cadence is under test. The mapping-fit month is deliberately
    drawn from a later, shifted distribution than the models' own training
    data, so a stale mapping has a genuine population shift to fail on.
    """
    train_df_old = _synthetic_baf_like(6000, seed=0, fraud_rate=0.02, month=0)
    train_df_new = pd.concat(
        [train_df_old, _synthetic_baf_like(3000, seed=1, fraud_rate=0.02, month=0)], ignore_index=True
    )
    # A later month with a shifted signal distribution (larger separation
    # between fraud/legit), so the model's raw score distribution genuinely
    # moves between the models' training data and the mapping-fit/eval
    # months -- otherwise everything is already so close to identically
    # distributed that a stale vs. fresh mapping cannot be told apart.
    mapping_fit_df = _synthetic_baf_like(4000, seed=2, fraud_rate=0.02, month=1, signal_scale=2.5)
    eval_df = _synthetic_baf_like(4000, seed=3, fraud_rate=0.02, month=2, signal_scale=2.5)

    old_model = train_model(train_df_old, beta=0.3, seed=0, n_estimators=80)
    new_model = train_model(train_df_new, beta=0.3, seed=0, n_estimators=80)
    mapping_old = QuantileMapping.fit(old_model.corrected_scores(train_df_new))
    # mapping_default must be a DISTINCT mapping from mapping_old -- see
    # run_model_update_experiment's docstring for why there is no
    # implicit fallback for this parameter.
    mapping_default = QuantileMapping.fit(new_model.corrected_scores(train_df_old))

    exp = run_model_update_experiment(
        old_model, new_model, mapping_fit_df, eval_df,
        fit_window_label="fit", eval_window_label="eval",
        mapping_old=mapping_old,
        mapping_default=mapping_default,
    )

    rel_err = exp.relative_error_table()
    err_by_predictor = rel_err.groupby("predictor")["relative_error_pct"].apply(lambda s: s.abs().median())

    # p2 must be at least as close to the target as p1_5.
    assert err_by_predictor["p2"] <= err_by_predictor["p1_5"] + 1e-9


def test_fixed_threshold_table_has_one_row_per_predictor():
    train_df_old = _synthetic_baf_like(1500, seed=0, fraud_rate=0.05, month=0)
    train_df_new = _synthetic_baf_like(1500, seed=1, fraud_rate=0.05, month=0)
    mapping_fit_df = _synthetic_baf_like(800, seed=2, fraud_rate=0.05, month=1)
    eval_df = _synthetic_baf_like(800, seed=3, fraud_rate=0.05, month=2)
    old_model = train_model(train_df_old, seed=0, n_estimators=20)
    new_model = train_model(train_df_new, seed=0, n_estimators=20)
    mapping_old = QuantileMapping.fit(old_model.corrected_scores(train_df_new))
    mapping_default = QuantileMapping.fit(new_model.corrected_scores(train_df_old))
    exp = run_model_update_experiment(
        old_model, new_model, mapping_fit_df, eval_df,
        fit_window_label="fit", eval_window_label="eval",
        mapping_old=mapping_old,
        mapping_default=mapping_default,
    )
    table = exp.fixed_threshold_table(edge=0.5)
    assert set(table["predictor"]) == {"p1", "p1_5", "p2", "custom", "default", "raw"}
    assert (table["alert_rate"] >= 0).all() and (table["alert_rate"] <= 1).all()


def test_run_model_update_experiment_requires_mapping_default():
    """mapping_default must be a REQUIRED argument with no implicit
    fallback to mapping_old -- an earlier version silently defaulted it
    to mapping_old, which made `default` an exact duplicate of `p1_5`
    for any caller (direct or indirect) that omitted it. Omitting it now
    must raise, not silently produce a degenerate `default` predictor."""
    train_df_old = _synthetic_baf_like(1500, seed=0, fraud_rate=0.05, month=0)
    train_df_new = _synthetic_baf_like(1500, seed=1, fraud_rate=0.05, month=0)
    mapping_fit_df = _synthetic_baf_like(800, seed=2, fraud_rate=0.05, month=1)
    eval_df = _synthetic_baf_like(800, seed=3, fraud_rate=0.05, month=2)
    old_model = train_model(train_df_old, seed=0, n_estimators=20)
    new_model = train_model(train_df_new, seed=0, n_estimators=20)
    mapping_old = QuantileMapping.fit(old_model.corrected_scores(train_df_new))
    with pytest.raises(TypeError):
        run_model_update_experiment(
            old_model, new_model, mapping_fit_df, eval_df,
            fit_window_label="fit", eval_window_label="eval",
            mapping_old=mapping_old,
        )


def test_drift_baseline_table_is_zero_drift_when_fit_and_eval_are_identical():
    """If the eval window IS the mapping's own fit window, there is by
    definition no temporal drift: p1's in-window alert rate and its
    held-out alert rate must be identical, even though both may carry a
    nonzero error against the target (a real fitting/sampling artifact,
    not drift)."""
    train_df_old = _synthetic_baf_like(1500, seed=0, fraud_rate=0.05, month=0)
    train_df_new = _synthetic_baf_like(1500, seed=1, fraud_rate=0.05, month=0)
    mapping_fit_df = _synthetic_baf_like(800, seed=2, fraud_rate=0.05, month=1)
    old_model = train_model(train_df_old, seed=0, n_estimators=20)
    new_model = train_model(train_df_new, seed=0, n_estimators=20)
    mapping_old = QuantileMapping.fit(old_model.corrected_scores(mapping_fit_df))
    mapping_default = QuantileMapping.fit(new_model.corrected_scores(train_df_old))
    exp = run_model_update_experiment(
        old_model, new_model, mapping_fit_df, mapping_fit_df,
        fit_window_label="fit", eval_window_label="fit (same)",
        mapping_old=mapping_old,
        mapping_default=mapping_default,
    )
    drift = exp.drift_baseline_table(mapping_fit_df)
    np.testing.assert_allclose(
        drift["fit_window_alert_rate"].to_numpy(),
        drift["eval_window_alert_rate"].to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        drift["fit_window_relative_error_pct"].to_numpy(),
        drift["temporal_drift_relative_error_pct"].to_numpy(),
        atol=1e-9,
    )


def test_bootstrap_relative_error_table_shape_and_all_predictors_measured():
    train_df_old = _synthetic_baf_like(1500, seed=0, fraud_rate=0.05, month=0)
    train_df_new = _synthetic_baf_like(1500, seed=1, fraud_rate=0.05, month=0)
    mapping_fit_df = _synthetic_baf_like(800, seed=2, fraud_rate=0.05, month=1)
    eval_df = _synthetic_baf_like(800, seed=3, fraud_rate=0.05, month=2)
    old_model = train_model(train_df_old, seed=0, n_estimators=20)
    new_model = train_model(train_df_new, seed=0, n_estimators=20)
    mapping_old = QuantileMapping.fit(old_model.corrected_scores(train_df_new))
    mapping_default = QuantileMapping.fit(new_model.corrected_scores(train_df_old))
    exp = run_model_update_experiment(
        old_model, new_model, mapping_fit_df, eval_df,
        fit_window_label="fit", eval_window_label="eval",
        mapping_old=mapping_old,
        mapping_default=mapping_default,
    )
    boot = exp.bootstrap_relative_error_table(n_bootstrap=20, seed=0)
    assert set(boot["predictor"]) == {"p1", "p1_5", "p2", "custom", "default", "raw"}
    assert boot["bootstrap_iter"].nunique() == 20
    # All predictors are measured against the same fixed target bin density,
    # so none of them should be identically zero here: this is real data
    # with real sampling noise, not a value that is zero by construction.
    for name in ("p1", "p1_5", "p2"):
        errors = boot.loc[boot["predictor"] == name, "relative_error_pct"].dropna()
        assert (errors.abs() > 0).any(), f"{name}'s relative error should not be trivially zero"


def test_walk_forward_experiment_produces_one_point_per_eval_month():
    df = pd.concat(
        [_synthetic_baf_like(1200, seed=m, fraud_rate=0.05, month=m) for m in range(6)],
        ignore_index=True,
    )
    rel_err = run_walk_forward_experiment(
        df,
        model_train_months=[0, 1], eval_months=[3, 4, 5],
        n_estimators_old=20, n_estimators_new=20, seed=0,
    )
    assert set(rel_err["point_idx"].unique()) == {0, 1, 2}
    assert list(rel_err.groupby("point_idx")["mapping_new_fit_month"].first()) == [2, 3, 4]
    assert list(rel_err.groupby("point_idx")["eval_month"].first()) == [3, 4, 5]


def test_walk_forward_experiment_rejects_when_no_eval_months_remain():
    df = pd.concat(
        [_synthetic_baf_like(1200, seed=m, fraud_rate=0.05, month=m) for m in range(3)],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        run_walk_forward_experiment(
            df, model_train_months=[0, 1], eval_months=[],
            n_estimators_old=20, n_estimators_new=20, seed=0,
        )


def test_walk_forward_experiment_rejects_eval_months_overlapping_training():
    """eval_months may not overlap model_train_months or the initial
    mapping-fit month -- otherwise a predictor would be scored on data its
    own model or mapping already saw."""
    df = pd.concat(
        [_synthetic_baf_like(1200, seed=m, fraud_rate=0.05, month=m) for m in range(6)],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        run_walk_forward_experiment(
            df, model_train_months=[0, 1], eval_months=[1, 3, 4],
            n_estimators_old=20, n_estimators_new=20, seed=0,
        )
    with pytest.raises(ValueError):
        run_walk_forward_experiment(
            df, model_train_months=[0, 1], eval_months=[2, 3, 4],  # 2 is the mapping-fit month
            n_estimators_old=20, n_estimators_new=20, seed=0,
        )


def test_walk_forward_experiment_default_uses_its_own_frozen_mapping():
    """The `default` variant must use its OWN dedicated "fit once, never
    revisited" mapping (`mapping_default`), fit on a month distinct from
    mapping_old (p1/p1_5), mapping_new (p2), and mapping_custom -- so
    `default` is never numerically identical to `p1_5`, `p2`, or `custom`
    at any evaluation step, including the first one. In particular,
    `default` must NOT collapse to an exact duplicate of `p1_5` (both
    score new_model; the only thing that can tell them apart is which
    mapping each one uses -- if that mapping were the same, they would be
    indistinguishable predictors, which defeats the point of including
    both)."""
    df = pd.concat(
        [_synthetic_baf_like(1200, seed=m, fraud_rate=0.05, month=m, signal_scale=1.0 + 0.3 * m) for m in range(6)],
        ignore_index=True,
    )
    rel_err = run_walk_forward_experiment(
        df,
        model_train_months=[0, 1], eval_months=[3, 4, 5],
        # a real hyperparameter gap between old/new is required for their
        # mappings (and therefore default vs. p1_5/p2/custom) to differ
        # at all -- with identical hyperparameters old_model/new_model
        # would be near-identical models and their mappings would
        # coincide too.
        beta_old=0.10, beta_new=0.80, n_estimators_old=10, n_estimators_new=60, seed=0,
    )
    for point_idx in rel_err["point_idx"].unique():
        step = rel_err[rel_err["point_idx"] == point_idx]
        default_err = step[step["predictor"] == "default"]["alert_rate"].to_numpy()
        p1_5_err = step[step["predictor"] == "p1_5"]["alert_rate"].to_numpy()
        p2_err = step[step["predictor"] == "p2"]["alert_rate"].to_numpy()
        custom_err = step[step["predictor"] == "custom"]["alert_rate"].to_numpy()
        assert not np.allclose(default_err, p1_5_err), (
            f"default and p1_5 are identical at point_idx={point_idx} -- default must use its own "
            "mapping_default, not mapping_old (which would make it an exact duplicate of p1_5)"
        )
        assert not np.allclose(default_err, p2_err), (
            f"default and p2 are identical at point_idx={point_idx}, but they use "
            "different mappings (mapping_default vs mapping_new) and should not coincide"
        )
        assert not np.allclose(default_err, custom_err), (
            f"default and custom are identical at point_idx={point_idx}, but they use "
            "different mappings (mapping_default vs mapping_custom) and should not coincide"
        )
        assert len(default_err) == len(p1_5_err) == len(p2_err) == len(custom_err)


def test_custom_mapping_diverges_from_p2_at_every_step_including_the_first():
    """custom's mapping is fit on a WIDER window than p2's single month at
    EVERY evaluation step, including the first one (the window always
    includes mapping_old_fit_month, not just months strictly after it) --
    so their bucket assignments (and therefore relative errors) are not
    required to be identical at any step, unlike a naive `custom = alias
    of p2` implementation where every row would match p2's exactly by
    construction."""
    df = pd.concat(
        [_synthetic_baf_like(1200, seed=m, fraud_rate=0.05, month=m, signal_scale=1.0 + 0.3 * m) for m in range(6)],
        ignore_index=True,
    )
    rel_err = run_walk_forward_experiment(
        df,
        model_train_months=[0, 1], eval_months=[3, 4, 5],
        n_estimators_old=20, n_estimators_new=20, seed=0,
    )
    for point_idx in rel_err["point_idx"].unique():
        step = rel_err[rel_err["point_idx"] == point_idx]
        p2_err = step[step["predictor"] == "p2"]["relative_error_pct"].to_numpy()
        custom_err = step[step["predictor"] == "custom"]["relative_error_pct"].to_numpy()
        assert not np.allclose(p2_err, custom_err), (
            f"custom and p2 should not be numerically identical at point_idx={point_idx}"
        )
