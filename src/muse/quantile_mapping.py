"""Quantile mapping fit against a target alert-rate curve.

A quantile mapping is a set of score thresholds, one per bin edge, chosen so
that a *reference* population's scores land in each bin at a pre-specified
target rate. Given a business-defined target curve

    target_alert_rates[e] = P(mapped_score >= e)   for e in (0, 1)

the mapping is fit by taking the (1 - target_alert_rates[e]) percentile of a
sample of raw scores for every edge e. Applying the mapping to *any* other
sample of raw scores (via `width_bucket`-style binning) then measures how
close that sample's alert rate is to the same fixed target -- which is
exactly what a fixed-threshold alerting rule needs to stay stable as the
model producing the raw scores changes over time.
"""

from __future__ import annotations

import numpy as np

# Cumulative target alert rate at each bin edge: target_alert_rates[e] is the
# fraction of events expected to score at or above e once the mapping is
# applied. 0.0 -> 1.0 (everything alerts at the lowest edge) and 1.0 is
# implicitly 0.0 (nothing scores above the top edge).
DEFAULT_TARGET_ALERT_RATES: dict[float, float] = {
    0.0: 1.0,
    0.2: 0.70,
    0.5: 0.35,
    0.6: 0.10,
    0.8: 0.05,
}

BIN_EDGES = [0.0, 0.2, 0.5, 0.6, 0.8, 1.0]


class QuantileMapping:
    """A fitted set of score thresholds, one per interior bin edge.

    `thresholds` are the raw-score cut points such that, on the sample the
    mapping was fit on, exactly `target_alert_rates[edge]` of events score at
    or above the threshold for that edge.
    """

    def __init__(self, thresholds: np.ndarray, target_alert_rates: dict[float, float]):
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.target_alert_rates = dict(target_alert_rates)

    @classmethod
    def fit(cls, scores: np.ndarray, target_alert_rates: dict[float, float] | None = None) -> "QuantileMapping":
        """Fit one threshold per interior edge (excludes 0.0 and 1.0, which
        need no threshold) from a sample of raw scores.
        """
        if target_alert_rates is None:
            target_alert_rates = DEFAULT_TARGET_ALERT_RATES
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            raise ValueError("cannot fit quantile mapping on an empty sample")

        interior_edges = sorted(e for e in target_alert_rates if e not in (0.0, 1.0))
        percentiles = [1.0 - target_alert_rates[e] for e in interior_edges]
        thresholds = np.quantile(scores, percentiles)
        return cls(thresholds, target_alert_rates)

    def bin_edges_in_score_space(self) -> np.ndarray:
        """Full bin-edge array in raw-score space: [0.0, t1, ..., tk, 1.0]."""
        return np.concatenate([[0.0], self.thresholds, [1.0]])

    def bucket(self, scores: np.ndarray) -> np.ndarray:
        """Bucket index (0-based) of each score under this mapping's edges."""
        scores = np.asarray(scores, dtype=float)
        edges = self.bin_edges_in_score_space()
        # searchsorted gives, for each score, how many edges[1:-1] it is >=,
        # which is exactly its 0-based bucket index (clipped to stay in range
        # for scores at or beyond the extremes).
        idx = np.searchsorted(edges[1:-1], scores, side="right")
        return np.clip(idx, 0, len(edges) - 2)

    def alert_rate(self, scores: np.ndarray, edge: float) -> float:
        """Observed fraction of `scores` at or beyond the score-space
        threshold fitted for cumulative bin edge `edge`.
        """
        interior_edges = sorted(e for e in self.target_alert_rates if e not in (0.0, 1.0))
        scores = np.asarray(scores, dtype=float)
        if edge == 0.0:
            return 1.0
        if edge not in interior_edges:
            raise ValueError(f"edge {edge} was not one of the edges this mapping was fit for")
        score_threshold = self.thresholds[interior_edges.index(edge)]
        return float(np.mean(scores >= score_threshold))
