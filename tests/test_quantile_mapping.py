"""Self-checks for the quantile mapping and posterior correction transforms.

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from muse.quantile_mapping import (
    BIN_EDGES,
    DEFAULT_TARGET_ALERT_RATES,
    QuantileMapping,
)
from muse.posterior_correction import posterior_correction


# ---------------------------------------------------------------------------
# QuantileMapping
# ---------------------------------------------------------------------------


def test_fit_produces_one_threshold_per_interior_edge():
    rng = np.random.default_rng(0)
    scores = rng.beta(0.5, 5, size=5000)
    qm = QuantileMapping.fit(scores)
    n_interior_edges = len([e for e in DEFAULT_TARGET_ALERT_RATES if e not in (0.0, 1.0)])
    assert len(qm.thresholds) == n_interior_edges


def test_fit_thresholds_are_non_decreasing():
    rng = np.random.default_rng(0)
    scores = rng.beta(0.5, 5, size=5000)
    qm = QuantileMapping.fit(scores)
    assert np.all(np.diff(qm.thresholds) >= 0)


def test_fit_reproduces_target_alert_rate_on_its_own_fit_sample():
    """Applying the mapping back to the exact sample it was fit on should
    reproduce the target alert rate at each edge almost exactly (up to
    quantile-interpolation rounding)."""
    rng = np.random.default_rng(0)
    scores = rng.beta(0.5, 5, size=20000)
    qm = QuantileMapping.fit(scores)

    for edge, target in DEFAULT_TARGET_ALERT_RATES.items():
        if edge in (0.0, 1.0):
            continue
        observed = qm.alert_rate(scores, edge)
        assert observed == pytest.approx(target, abs=0.01)


def test_bucket_indices_are_within_range():
    rng = np.random.default_rng(0)
    scores = rng.beta(0.5, 5, size=2000)
    qm = QuantileMapping.fit(scores)
    buckets = qm.bucket(rng.uniform(0, 1, size=500))
    assert buckets.min() >= 0
    assert buckets.max() <= len(BIN_EDGES) - 2


def test_rejects_empty_sample():
    with pytest.raises(ValueError):
        QuantileMapping.fit(np.array([]))


def test_handles_degenerate_ties_without_crashing():
    """A window with a large constant-score cluster must not raise -- the
    mapping should still fit and bucket without crashing."""
    scores = np.concatenate([np.zeros(500), np.array([0.9, 0.95, 0.99])])
    qm = QuantileMapping.fit(scores)
    buckets = qm.bucket(np.array([0.0, 0.5, 0.99]))
    assert np.all(np.isfinite(buckets))


def test_alert_rate_rejects_unknown_edge():
    rng = np.random.default_rng(0)
    scores = rng.beta(0.5, 5, size=1000)
    qm = QuantileMapping.fit(scores)
    with pytest.raises(ValueError):
        qm.alert_rate(scores, edge=0.55)


# ---------------------------------------------------------------------------
# Posterior correction
# ---------------------------------------------------------------------------


def test_posterior_correction_identity_at_beta_one():
    scores = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(posterior_correction(scores, beta=1.0), scores)


def test_posterior_correction_reduces_score_under_undersampling():
    """With beta < 1 (majority class undersampled), raw scores are inflated;
    correction should always pull them back down (or leave 0/1 fixed)."""
    scores = np.array([0.1, 0.5, 0.9])
    corrected = posterior_correction(scores, beta=0.1)
    assert np.all(corrected <= scores)


def test_posterior_correction_rejects_invalid_beta():
    with pytest.raises(ValueError):
        posterior_correction(np.array([0.5]), beta=0.0)
    with pytest.raises(ValueError):
        posterior_correction(np.array([0.5]), beta=1.5)
