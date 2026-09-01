"""End-to-end run of the model-update alert-rate-stability study on BAF.

Trains old/new models once, walks forward over BAF's remaining months
evaluating all six predictor variants at each step (see
muse.experiment's module docstring for the design and
predictor definitions). Concatenates every walk-forward step's
relative-error table across all seeds in `--seeds` into one CSV/set of
plots per `--variant`, so the box plots show spread across both steps and
seeds, not just one seed -- a check that the staleness effect isn't a
single-seed artifact. `--variant` accepts multiple BAF variants (Base
plus any of the bias/fairness Variant I-V) to check the effect holds
under BAF's demographic-shift conditions too.

Usage:
    uv run python scripts/run_experiment.py --variant Base --seeds 0
    uv run python scripts/run_experiment.py --variant Base "Variant II" --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from muse.data import VARIANTS, load_variant
from muse.experiment import (
    fixed_threshold_table,
    run_model_update_experiment,
    run_walk_forward_experiment,
)
from muse.models import train_model
from muse.plotting import plot_fixed_threshold_bars, plot_relative_error_boxplot, set_paper_style
from muse.quantile_mapping import QuantileMapping

MODEL_UPDATE_PREDICTORS = ["p1", "p1_5", "p2"]
CALIBRATION_PREDICTORS = ["raw", "default", "custom"]
FIXED_EDGE = 0.6

# Default old/model LightGBM training params
BETA_OLD = 0.05
BETA_NEW = 0.50
N_ESTIMATORS_OLD = 20
N_ESTIMATORS_NEW = 500
MAX_DEPTH_OLD = 1
MAX_DEPTH_NEW = 10
NUM_LEAVES_OLD = 2
NUM_LEAVES_NEW = 128
LEARNING_RATE_OLD = 0.5
LEARNING_RATE_NEW = 0.03


def _run_one(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[int], list[int], int]:
    """Run the walk-forward experiment once for a given (already-loaded)
    variant DataFrame and seed. Returns (relative_error_df, months,
    model_train_months, mapping_fit_month)."""
    months = sorted(int(m) for m in df["month"].unique())
    model_train_months = months[:2]
    mapping_fit_month = max(model_train_months) + 1
    eval_months = [m for m in months if m > mapping_fit_month]

    rel_err_df = run_walk_forward_experiment(
        df,
        model_train_months=model_train_months,
        eval_months=eval_months,
        beta_old=BETA_OLD,
        beta_new=BETA_NEW,
        n_estimators_old=N_ESTIMATORS_OLD,
        n_estimators_new=N_ESTIMATORS_NEW,
        max_depth_old=MAX_DEPTH_OLD,
        max_depth_new=MAX_DEPTH_NEW,
        num_leaves_old=NUM_LEAVES_OLD,
        num_leaves_new=NUM_LEAVES_NEW,
        learning_rate_old=LEARNING_RATE_OLD,
        learning_rate_new=LEARNING_RATE_NEW,
        seed=seed,
    )
    return rel_err_df, months, model_train_months, mapping_fit_month


def _fixed_threshold_snapshot(
    df: pd.DataFrame, months: list[int], model_train_months: list[int], mapping_fit_month: int, seed: int
):
    """Re-derive one walk-forward step's PredictorResults (the LAST
    evaluation month, for the given seed) purely to plot fixed-threshold
    alert rates -- run_walk_forward_experiment only returns the aggregated
    relative-error table, not the underlying PredictorResult objects."""
    train_df = df[df["month"].isin(model_train_months)].reset_index(drop=True)
    old_model = train_model(
        train_df, beta=BETA_OLD, seed=seed, n_estimators=N_ESTIMATORS_OLD,
        max_depth=MAX_DEPTH_OLD, num_leaves=NUM_LEAVES_OLD, learning_rate=LEARNING_RATE_OLD,
    )
    new_model = train_model(
        train_df, beta=BETA_NEW, seed=seed, n_estimators=N_ESTIMATORS_NEW,
        max_depth=MAX_DEPTH_NEW, num_leaves=NUM_LEAVES_NEW, learning_rate=LEARNING_RATE_NEW,
    )
    mapping_old_fit_df = df[df["month"] == mapping_fit_month].reset_index(drop=True)
    mapping_old = QuantileMapping.fit(old_model.corrected_scores(mapping_old_fit_df))
    mapping_default_fit_df = df[df["month"] == min(model_train_months)].reset_index(drop=True)
    mapping_default = QuantileMapping.fit(new_model.corrected_scores(mapping_default_fit_df))

    last_eval_month = max(m for m in months if m > mapping_fit_month)
    mapping_new_fit_df = df[df["month"] == last_eval_month - 1].reset_index(drop=True)
    eval_df = df[df["month"] == last_eval_month].reset_index(drop=True)
    return run_model_update_experiment(
        old_model=old_model, new_model=new_model,
        mapping_new_fit_df=mapping_new_fit_df, eval_df=eval_df,
        fit_window_label=f"month {last_eval_month - 1}", eval_window_label=f"month {last_eval_month}",
        mapping_old=mapping_old,
        mapping_default=mapping_default,
    )


def _run_variant(variant: str, data_dir: str | None, seeds: list[int], output_dir: Path) -> dict:
    print(f"\n=== Variant '{variant}' ===")
    print(f"Loading BAF variant '{variant}'...")
    df = load_variant(variant, data_dir=data_dir)

    variant_dir = output_dir / variant.replace(" ", "_")
    variant_dir.mkdir(parents=True, exist_ok=True)

    per_seed_median: dict[str, list[float]] = {}
    months = model_train_months = mapping_fit_month = None
    rel_err_df_by_seed: dict[int, pd.DataFrame] = {}

    for seed in seeds:
        rel_err_df, months, model_train_months, mapping_fit_month = _run_one(df, seed)
        rel_err_df_by_seed[seed] = rel_err_df
        median_abs_err = rel_err_df.groupby("predictor")["relative_error_pct"].apply(lambda s: s.abs().median())
        for name, val in median_abs_err.items():
            per_seed_median.setdefault(name, []).append(float(val))
        print(f"  seed={seed}: " + ", ".join(f"{n}={v:.1f}%" for n, v in sorted(median_abs_err.items())))

    # Concatenate every seed's per-step rows (tagged with a "seed" column)
    # into one CSV/one set of plots per variant -- the box plots then show
    # spread across BOTH walk-forward steps and seeds, not just one seed.
    combined = []
    for seed, per_seed_df in rel_err_df_by_seed.items():
        tagged = per_seed_df.copy()
        tagged["seed"] = seed
        combined.append(tagged)
    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(variant_dir / "relative_error.csv", index=False)

    seed_summary = {
        name: {
            "median": statistics.median(vals),
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "per_seed": dict(zip(seeds, vals)),
        }
        for name, vals in per_seed_median.items()
    }
    summary = {
        "variant": variant,
        "seeds": seeds,
        "months": months,
        "model_train_months": model_train_months,
        "mapping_fit_month": mapping_fit_month,
        "n_eval_steps": int(combined_df["point_idx"].nunique()),
        "beta_old": BETA_OLD,
        "beta_new": BETA_NEW,
        "n_estimators_old": N_ESTIMATORS_OLD,
        "n_estimators_new": N_ESTIMATORS_NEW,
        "max_depth_old": MAX_DEPTH_OLD,
        "max_depth_new": MAX_DEPTH_NEW,
        "num_leaves_old": NUM_LEAVES_OLD,
        "num_leaves_new": NUM_LEAVES_NEW,
        "learning_rate_old": LEARNING_RATE_OLD,
        "learning_rate_new": LEARNING_RATE_NEW,
        "fixed_edge": FIXED_EDGE,
        # Across-seed distribution of each predictor's median |relative
        # error| -- the honest replacement for a single seed=0 point
        # estimate (see docstring above / README note on seed variance).
        "median_abs_relative_error_pct_by_predictor_across_seeds": seed_summary,
    }
    with open(variant_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Multiple plots per variant: the two relative-error box plots
    # (model-update predictors, then calibration-only predictors) plus a
    # fixed-threshold alert-rate bar chart, all scoped to this variant.
    set_paper_style()
    fig1 = plot_relative_error_boxplot(
        combined_df, predictors=MODEL_UPDATE_PREDICTORS,
        title=f"Alert-rate relative error, per predictor -- {variant}",
    )
    fig1.savefig(variant_dir / "relative_error_boxplot.pdf")
    fig2 = plot_relative_error_boxplot(
        combined_df, predictors=CALIBRATION_PREDICTORS,
        title=f"Alert-rate relative error, calibration-only (fixed model) -- {variant}",
    )
    fig2.savefig(variant_dir / "relative_error_boxplot_calibration.pdf")

    last_step = _fixed_threshold_snapshot(df, months, model_train_months, mapping_fit_month, seeds[-1])
    fig3 = plot_fixed_threshold_bars(
        fixed_threshold_table(last_step.predictors, edge=FIXED_EDGE),
        edge=FIXED_EDGE,
        predictors=MODEL_UPDATE_PREDICTORS + CALIBRATION_PREDICTORS,
    )
    fig3.savefig(variant_dir / "fixed_threshold_bars.pdf")

    print(f"\n  Median |relative error| by predictor, across seeds {seeds} (%):")
    for name, stats in sorted(seed_summary.items()):
        print(
            f"    {name}: median={stats['median']:.1f}  mean={stats['mean']:.1f}  "
            f"stdev={stats['stdev']:.1f}  range=[{stats['min']:.1f}, {stats['max']:.1f}]"
        )
    print(f"  Results written to {variant_dir}/")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", nargs="+", default=["Base"], choices=VARIANTS)
    parser.add_argument("--data-dir", default=None, help="Local dir with extracted BAF CSVs")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for variant in args.variant:
        all_summaries[variant] = _run_variant(variant, args.data_dir, args.seeds, output_dir)

    with open(output_dir / "summary_all_variants.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll variants done. Combined summary at {output_dir}/summary_all_variants.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
