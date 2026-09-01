"""Model training for the model-update experiment.

Simulates a model replacement: swapping a deployed classifier for a new one
trained on the same data and feature set, but with very different capacity
(a shallow, under-tuned legacy model replaced by a properly tuned one --
different max_depth/num_leaves/learning_rate/n_estimators/undersampling
ratio, all on the exact same training rows). This produces a genuine shift
in the raw score distribution -- the same kind of shift that happens
whenever a production model is retrained or upgraded -- without
confounding the comparison with a change in training data or training
recency.

Models are trained with majority-class undersampling, a common practice in
imbalanced classification, so the resulting scores are realistically skewed.
Posterior correction is applied before quantile mapping, mirroring a
two-stage scoring pipeline: score = quantile_mapping(posterior_correction(model(x))).

LightGBM training is pinned to single-threaded, deterministic, row-wise
histogram construction (see train_model()) so a given seed reproduces
bit-for-bit identical trees regardless of the host machine's core count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder

from muse.data import feature_columns
from muse.posterior_correction import posterior_correction

CATEGORICAL_COLUMNS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]


@dataclass
class TrainedModel:
    """A model trained with majority-class undersampling, plus its beta
    (kept negative-class fraction) needed for posterior correction."""

    model: LGBMClassifier
    encoder: OrdinalEncoder
    beta: float
    feature_names: list[str]

    def raw_scores(self, df: pd.DataFrame) -> np.ndarray:
        x = _encode_features(df, self.encoder, self.feature_names)
        return self.model.predict_proba(x)[:, 1]

    def corrected_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Raw model score after posterior correction -- the score that
        would be passed into the quantile mapping stage."""
        raw = self.raw_scores(df)
        return posterior_correction(raw, self.beta)


def _encode_features(df: pd.DataFrame, encoder: OrdinalEncoder, feature_names: list[str]) -> pd.DataFrame:
    x = df[feature_names].copy()
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_names]
    if cat_cols:
        x[cat_cols] = encoder.transform(x[cat_cols])
    return x


def undersample_majority(df: pd.DataFrame, beta: float, seed: int) -> pd.DataFrame:
    """Keep all fraud rows, keep a `beta` fraction of legitimate rows.

    beta is the majority-class retention ratio used by posterior correction:
    T(y; beta) = beta*y / (1 - (1-beta)*y).
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    rng = np.random.default_rng(seed)
    fraud = df[df["fraud_bool"] == 1]
    legit = df[df["fraud_bool"] == 0]
    if beta < 1.0:
        keep_mask = rng.random(len(legit)) < beta
        legit = legit.loc[keep_mask]
    return pd.concat([fraud, legit], ignore_index=True)


def train_model(
    df: pd.DataFrame,
    beta: float = 0.10,
    seed: int = 0,
    n_estimators: int = 200,
    max_depth: int = -1,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
) -> TrainedModel:
    """Train a LightGBM classifier on BAF rows with majority-class
    undersampling ratio `beta`, using all standard BAF features.

    `max_depth`/`num_leaves`/`learning_rate` default to LightGBM's own
    defaults; the walk-forward experiment deliberately sets these very
    differently for old_model vs. new_model (a shallow, under-tuned
    legacy model vs. a properly tuned one) so the two models' raw score
    distributions diverge enough for a mapping-staleness comparison to be
    meaningful, even though both train on the exact same data (see
    run_experiment.py and run_walk_forward_experiment's module docstring).
    """
    feature_names = feature_columns(df)
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_names]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(df[cat_cols])

    train_df = undersample_majority(df, beta=beta, seed=seed)
    x_train = _encode_features(train_df, encoder, feature_names)
    y_train = train_df["fraud_bool"].to_numpy()

    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        random_state=seed,
        verbosity=-1,
        # LightGBM's histogram building can still be non-deterministic
        # across machines/thread counts with random_state alone; pin
        # single-threaded, deterministic, row-wise histogram construction
        # so a given seed reproduces bit-for-bit identical trees regardless
        # of the host machine's core count.
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    return TrainedModel(model=model, encoder=encoder, beta=beta, feature_names=feature_names)
