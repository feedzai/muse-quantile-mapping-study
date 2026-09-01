"""Posterior correction for undersampled binary classifiers.

    T(y; beta) = (beta * y) / (1 - (1 - beta) * y)

When training a classifier on an imbalanced dataset, it's common to
undersample the majority (negative) class to speed up training and improve
learning efficiency. This shifts the model's predicted scores upward
relative to what they would be on the true, non-resampled class balance.
Posterior correction reverses that shift analytically, given only the
undersampling ratio `beta` (the fraction of the majority class kept during
training) -- no labeled evaluation data is required.
"""

from __future__ import annotations

import numpy as np


def posterior_correction(scores: np.ndarray, beta: float) -> np.ndarray:
    """Rescale raw scores from a model trained with majority-class
    undersampling ratio `beta` back to the original class-balance scale.

    beta=1.0 (no undersampling) is the identity transform.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    scores = np.asarray(scores, dtype=float)
    return (beta * scores) / (1.0 - (1.0 - beta) * scores)
