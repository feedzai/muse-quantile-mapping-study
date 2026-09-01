#!/usr/bin/env python3
"""Compute and print the old_model/new_model score correlation used to
justify run_experiment.py's hyperparameter gap.

The README and experiment.py's docstrings claim specific correlation
numbers (~0.7-0.8 depending on the hyperparameter gap) as the "checkable"
evidence that old_model and new_model genuinely diverge. This script is
that check: it trains old_model/new_model with the exact same
hyperparameters run_experiment.py uses, scores them on the mapping-fit
month, and prints their Pearson correlation, so the number in the README
is reproducible instead of asserted.

Usage:
    uv run python scripts/check_model_divergence.py [--variant Base]
"""

from __future__ import annotations

import argparse

from muse.data import VARIANTS, load_variant
from muse.experiment import model_score_correlation
from muse.models import train_model
from run_experiment import (
    BETA_NEW,
    BETA_OLD,
    LEARNING_RATE_NEW,
    LEARNING_RATE_OLD,
    MAX_DEPTH_NEW,
    MAX_DEPTH_OLD,
    NUM_LEAVES_NEW,
    NUM_LEAVES_OLD,
    N_ESTIMATORS_NEW,
    N_ESTIMATORS_OLD,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="Base", choices=VARIANTS)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_variant(args.variant, data_dir=args.data_dir)
    months = sorted(int(m) for m in df["month"].unique())
    model_train_months = months[:2]
    mapping_fit_month = max(model_train_months) + 1

    train_df = df[df["month"].isin(model_train_months)].reset_index(drop=True)
    scoring_df = df[df["month"] == mapping_fit_month].reset_index(drop=True)

    old_model = train_model(
        train_df, beta=BETA_OLD, seed=args.seed, n_estimators=N_ESTIMATORS_OLD,
        max_depth=MAX_DEPTH_OLD, num_leaves=NUM_LEAVES_OLD, learning_rate=LEARNING_RATE_OLD,
    )
    new_model = train_model(
        train_df, beta=BETA_NEW, seed=args.seed, n_estimators=N_ESTIMATORS_NEW,
        max_depth=MAX_DEPTH_NEW, num_leaves=NUM_LEAVES_NEW, learning_rate=LEARNING_RATE_NEW,
    )
    corr = model_score_correlation(old_model, new_model, scoring_df)

    print(f"variant={args.variant} model_train_months={model_train_months} scoring_month={mapping_fit_month}")
    print(f"old_model params:  beta={BETA_OLD} n_estimators={N_ESTIMATORS_OLD} "
          f"max_depth={MAX_DEPTH_OLD} num_leaves={NUM_LEAVES_OLD} learning_rate={LEARNING_RATE_OLD}")
    print(f"new_model params:  beta={BETA_NEW} n_estimators={N_ESTIMATORS_NEW} "
          f"max_depth={MAX_DEPTH_NEW} num_leaves={NUM_LEAVES_NEW} learning_rate={LEARNING_RATE_NEW}")
    print(f"old_model vs new_model corrected-score correlation on month {mapping_fit_month}: {corr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
