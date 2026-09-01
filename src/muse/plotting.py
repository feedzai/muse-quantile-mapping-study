"""Plotting utilities for the model-update stability experiment.

Box plots of alert-rate relative error on a single symlog axis (linear
inside +/-100%, logarithmic beyond) -- no broken axes, since a
stale-mapping predictor can be off by orders of magnitude at some
thresholds while every other predictor stays close to 0% everywhere. Uses
a colorblind-safe palette (Okabe-Ito) plus a distinct hatch per predictor
so boxes stay distinguishable in grayscale/print.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from muse.experiment import EDGE_LABELS, N_EDGES

OKABE_ITO_ORANGE = "#E69F00"
OKABE_ITO_SKY_BLUE = "#56B4E9"
OKABE_ITO_GREEN = "#009E73"

PREDICTOR_STYLE = {
    "raw": {"color": OKABE_ITO_SKY_BLUE, "hatch": "//", "label": r"predictor $\mathit{raw}$"},
    "default": {"color": OKABE_ITO_ORANGE, "hatch": "xx", "label": r"predictor $\mathit{default}$"},
    "custom": {"color": OKABE_ITO_GREEN, "hatch": "..", "label": r"predictor $\mathit{custom}$"},
    "p1": {"color": OKABE_ITO_ORANGE, "hatch": "xx", "label": r"predictor $\mathit{p_1}$"},
    "p1_5": {"color": OKABE_ITO_SKY_BLUE, "hatch": "//", "label": r"predictor $\mathit{p_{1.5}}$"},
    "p2": {"color": OKABE_ITO_GREEN, "hatch": "..", "label": r"predictor $\mathit{p_2}$"},
}

LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 12

SYMLOG_TICKS = [-100, -50, 0, 50, 100, 500, 1000, 5000, 10000]
MIN_BOX_HEIGHT_FRAC = 0.10  # minimum rendered box height, as a fraction of linthresh


def set_paper_style() -> None:
    """A clean, publication-style matplotlib theme: serif/Computer Modern
    fonts and a light grid background."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.style.use("seaborn-v0_8-whitegrid")


def _binwise_observations(df: pd.DataFrame, predictors: list[str]) -> dict[tuple[str, int], np.ndarray]:
    obs = {}
    for predictor in predictors:
        sub = df[df["predictor"] == predictor]
        if sub.empty:
            continue
        for e in range(N_EDGES):
            cell = sub[sub["bucket_id"] == e]
            if cell.empty:
                continue
            vals = cell["relative_error_pct"].dropna().to_numpy()
            if len(vals):
                obs[(predictor, e)] = vals
    return obs


def plot_relative_error_boxplot(
    df: pd.DataFrame,
    predictors: list[str] | None = None,
    title: str = "Alert-rate relative error, per predictor",
    linthresh: float = 100,
) -> plt.Figure:
    """Box plots of alert-rate relative error, one box per predictor at
    each fixed-threshold edge, on a single symlog y axis.
    """
    if predictors is None:
        predictors = ["p1", "p1_5", "p2"]

    obs = _binwise_observations(df, predictors)
    if not obs:
        raise ValueError("no observations to plot")

    present = [p for p in predictors if any(k[0] == p for k in obs)]
    n_pred = len(present)
    group_w = 0.78
    box_w = group_w / max(n_pred, 1)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, predictor in enumerate(present):
        style = PREDICTOR_STYLE.get(predictor, {"color": "#777777", "hatch": None, "label": predictor})
        data, positions = [], []
        for e in range(N_EDGES):
            if (predictor, e) not in obs:
                continue
            data.append(obs[(predictor, e)])
            positions.append(e + (i - (n_pred - 1) / 2) * box_w)
        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_w * 0.86,
            patch_artist=True,
            showfliers=True,
            whis=(5, 95),
            medianprops=dict(color="black", linewidth=1.4),
            flierprops=dict(
                marker=".", markersize=3, markerfacecolor=style["color"], markeredgecolor="none", alpha=0.45
            ),
            capprops=dict(linewidth=1.0, color=style["color"]),
            whiskerprops=dict(linewidth=1.0, color=style["color"]),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(style["color"])
            patch.set_alpha(0.85)
            patch.set_edgecolor(style["color"])
            patch.set_linewidth(1.2)
            if style.get("hatch"):
                patch.set_hatch(style["hatch"])

        # A degenerate distribution (near-zero IQR) renders as an invisible
        # hairline on the symlog axis. Enforce a minimum rendered box height,
        # expanding away from the nearest data extreme so the widened box
        # doesn't get clipped by the axis limits.
        min_h = MIN_BOX_HEIGHT_FRAC * linthresh
        data_floor = min(v.min() for v in data)
        data_ceil = max(v.max() for v in data)
        for path in bp["boxes"]:
            verts = path.get_path().vertices
            y_lo, y_hi = verts[:, 1].min(), verts[:, 1].max()
            if (y_hi - y_lo) < min_h:
                y_mid = (y_hi + y_lo) / 2
                if (y_mid - data_floor) <= (data_ceil - y_mid):
                    new_lo, new_hi = y_lo, y_lo + min_h
                else:
                    new_lo, new_hi = y_hi - min_h, y_hi
                new_verts = verts.copy()
                new_verts[:, 1] = np.where(np.isclose(verts[:, 1], y_lo), new_lo, new_verts[:, 1])
                new_verts[:, 1] = np.where(np.isclose(verts[:, 1], y_hi), new_hi, new_verts[:, 1])
                path.get_path().vertices[:] = new_verts

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.8, zorder=1)
    for e in range(N_EDGES - 1):
        ax.axvline(e + 0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.45, zorder=0)

    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_yticks([t for t in SYMLOG_TICKS if t >= -100])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.axhspan(-linthresh, linthresh, color="grey", alpha=0.05, zorder=0)

    ax.set_xticks(range(N_EDGES))
    ax.set_xticklabels(EDGE_LABELS, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_xlim(-0.6, N_EDGES - 0.4)
    ax.set_xlabel("Fixed-Threshold Edge", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Relative Error, Alert Rate (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=AXIS_LABEL_FONTSIZE)

    handles = [
        mpatches.Patch(
            facecolor=PREDICTOR_STYLE.get(p, {}).get("color", "#777"),
            edgecolor=PREDICTOR_STYLE.get(p, {}).get("color", "#777"),
            alpha=0.85,
            hatch=PREDICTOR_STYLE.get(p, {}).get("hatch"),
            label=PREDICTOR_STYLE.get(p, {}).get("label", p),
        )
        for p in present
    ]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.08), frameon=False,
        fontsize=LEGEND_FONTSIZE, ncol=len(present),
    )

    fig.tight_layout()
    return fig


def plot_fixed_threshold_bars(
    fixed_threshold_df: pd.DataFrame, edge: float = 0.7, predictors: list[str] | None = None
) -> plt.Figure:
    """Bar chart of alert rate at a single fixed threshold edge, one bar
    per predictor."""
    if predictors is None:
        predictors = ["p1", "p1_5", "p2"]
    df = fixed_threshold_df[fixed_threshold_df["predictor"].isin(predictors)]
    df = df.set_index("predictor").loc[[p for p in predictors if p in df["predictor"].values]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [PREDICTOR_STYLE.get(p, {}).get("color", "#777") for p in df.index]
    hatches = [PREDICTOR_STYLE.get(p, {}).get("hatch") for p in df.index]
    bars = ax.bar(df.index, df["alert_rate"], color=colors, edgecolor="black", linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    ax.set_ylabel("Alert rate")
    ax.set_xlabel("Predictor")
    ax.set_title(f"Alert rate at fixed threshold edge {edge}")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
