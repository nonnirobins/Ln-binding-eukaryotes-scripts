#!/usr/bin/env python3
"""
PCA_classification.py
---------------------
Run PCA on (ΔcsRMSD, ΔΔE) for PQQ-dependent MDH structures, write PCA group
assignments back to the assessment TSV, and produce two figure panels:

  Panel A — ΔΔE vs ΔcsRMSD scatter coloured by binding label
  Panel B — PC1 strip plot coloured by binding label, with La-like /
             Transition / Ca-like zone boundaries

PCA group thresholds are set by the verified controls:
  La-like    — PC1 ≥ min(PC1 of Verified La Binding)
  Ca-like    — PC1 ≤ max(PC1 of Verified Ca Binding)
  Transition — between those boundaries

The columns  pca_group  and  pc1  are written back to the assessment TSV
so that Pairwise_csrmsd.py can colour the heatmap row annotations.

Rows are included in the PCA when:
  • quality_control == "kept"
  • qm_ddE_kcal is not NaN
  • qm_class does not contain "OUTLIER"
  • delta_csrmsd is not NaN

Usage:
  python PCA_classification.py
  python PCA_classification.py --assess /path/to/la_binding_assessment.tsv
  python PCA_classification.py --out-dir /path/to/output/

Dependencies: numpy, pandas, matplotlib, scipy, scikit-learn
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────
_BASE    = Path("/groups/banfield/projects/multienv/corkscrew/supplementary_structures")
_ASSESS  = _BASE / "la_binding_assessment.tsv"

# ── style constants ───────────────────────────────────────────────────────────
LABEL_ORDER = [
    "Verified La Binding", "Likely La Binding", "Novel La Binding",
    "Novel Ca Binding",    "Likely Ca Binding", "Verified Ca Binding",
    "Inconclusive",
]
LCOL = {
    "Verified La Binding": "#cc2222",
    "Likely La Binding":   "#e09a5a",
    "Novel La Binding":    "#e8c97a",
    "Novel Ca Binding":    "#9fa8da",
    "Likely Ca Binding":   "#1ecce8",
    "Verified Ca Binding": "#1a78c2",
    "Inconclusive":        "#aaaaaa",
}
MSTYLE = {
    "Verified La Binding": dict(marker="*", s=180, lw=0.8, zorder=5),
    "Verified Ca Binding": dict(marker="D", s=55,  lw=0.8, zorder=5),
    "Likely La Binding":   dict(marker="o", s=40,  lw=0.5, zorder=3),
    "Novel La Binding":    dict(marker="^", s=50,  lw=0.5, zorder=3),
    "Novel Ca Binding":    dict(marker="v", s=50,  lw=0.5, zorder=3),
    "Likely Ca Binding":   dict(marker="s", s=40,  lw=0.5, zorder=3),
    "Inconclusive":        dict(marker="X", s=40,  lw=0.5, zorder=3),
}
GRP_COL = {"La-like": "#e97132", "Transition": "#888888", "Ca-like": "#007d7d"}


def save_fig(path_stem):
    for ext in ("svg", "pdf"):
        plt.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight", format=ext)
        print(f"  Saved {path_stem}.{ext}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="PCA on (ΔcsRMSD, ΔΔE) and figure panels A+B for PQQ-MDH assessment."
    )
    ap.add_argument("--assess",  default=str(_ASSESS),
                    help="Assessment TSV (default: la_binding_assessment.tsv)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for figures (default: same dir as --assess)")
    args = ap.parse_args()

    assess_path = Path(args.assess)
    out_dir     = Path(args.out_dir) if args.out_dir else assess_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load & filter ─────────────────────────────────────────────────────────
    df_all  = pd.read_csv(assess_path, sep="\t")
    df_kept = df_all[df_all["quality_control"] == "kept"].copy()
    df_plot = df_kept[
        df_kept["qm_ddE_kcal"].notna() &
        ~df_kept["qm_class"].str.contains("OUTLIER", na=False) &
        df_kept["delta_csrmsd"].notna()
    ].copy()
    print(f"Proteins in scatter/PCA: {len(df_plot)}")

    # ── PCA ───────────────────────────────────────────────────────────────────
    X_scaled = StandardScaler().fit_transform(df_plot[["delta_csrmsd", "qm_ddE_kcal"]])
    pca      = PCA(n_components=2)
    scores   = pca.fit_transform(X_scaled)

    la_mask = df_plot["binding_label"] == "Verified La Binding"
    ca_mask = df_plot["binding_label"] == "Verified Ca Binding"
    if scores[la_mask, 0].mean() < scores[ca_mask, 0].mean():
        pca.components_ *= -1
        scores          *= -1

    df_plot["pc1"] = scores[:, 0]
    df_plot["pc2"] = scores[:, 1]
    la_pc1_min = df_plot.loc[la_mask, "pc1"].min()
    ca_pc1_max = df_plot.loc[ca_mask, "pc1"].max()

    df_plot["pca_group"] = np.select(
        [df_plot["pc1"] >= la_pc1_min, df_plot["pc1"] <= ca_pc1_max],
        ["La-like",                    "Ca-like"],
        default="Transition",
    )

    ev = pca.explained_variance_ratio_
    print(f"PC1 {ev[0]*100:.1f}%  PC2 {ev[1]*100:.1f}%  |  "
          f"La-like n={(df_plot['pca_group']=='La-like').sum()}  "
          f"Ca-like n={(df_plot['pca_group']=='Ca-like').sum()}  "
          f"Transition n={(df_plot['pca_group']=='Transition').sum()}")

    # write pca_group + pc1 back to TSV
    df_all = df_all.drop(columns=["pca_group", "pc1"], errors="ignore")
    df_all["pca_group"] = np.nan
    df_all["pc1"]       = np.nan
    df_all.loc[df_plot.index, "pca_group"] = df_plot["pca_group"]
    df_all.loc[df_plot.index, "pc1"]       = df_plot["pc1"].round(4)
    df_all.to_csv(assess_path, sep="\t", index=False)
    print("pca_group + pc1 written to TSV")

    # ── Panel A — ΔΔE vs ΔcsRMSD scatter ─────────────────────────────────────
    print("\nPanel A — scatter")
    x = df_plot["delta_csrmsd"].values
    y = df_plot["qm_ddE_kcal"].values
    slope, intercept, *_ = stats.linregress(x, y)
    rho, p_sp = stats.spearmanr(x, y)
    p_str = "p < 10⁻²⁰" if p_sp < 1e-20 else f"p = {p_sp:.2e}"

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for lbl in LABEL_ORDER:
        sub = df_plot[df_plot["binding_label"] == lbl]
        if sub.empty: continue
        ms = MSTYLE[lbl]
        ax.scatter(sub["delta_csrmsd"], sub["qm_ddE_kcal"],
                   marker=ms["marker"], s=ms["s"], linewidths=ms["lw"],
                   c=LCOL[lbl], edgecolors="white", alpha=0.88, zorder=ms["zorder"])

    x_fit = np.array([x.min() - 0.1, x.max() + 0.1])
    ax.plot(x_fit, slope * x_fit + intercept, color="#333333", lw=1.4, zorder=6, ls="--")
    ax.text(0.97, 0.04, f"ρ = {rho:.3f}\n{p_str}\nn = {len(df_plot)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8))
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.3, zorder=0)
    ax.set_xlabel("ΔcsRMSD (csRMSD_MxaF − csRMSD_XoxF)", fontsize=11)
    ax.set_ylabel("ΔΔE (La³⁺ vs Ca²⁺, kcal/mol)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.35)

    shape_handles = [
        Line2D([0], [0], marker=MSTYLE[lbl]["marker"], color="w",
               markerfacecolor=LCOL[lbl], markeredgecolor="white", markeredgewidth=0.5,
               markersize=9 if "Verified" in lbl else 6,
               label=f"{lbl} (n={len(df_plot[df_plot['binding_label']==lbl])})",
               linestyle="none")
        for lbl in LABEL_ORDER if not df_plot[df_plot["binding_label"] == lbl].empty
    ]
    ax.legend(handles=shape_handles, fontsize=7.5, loc="upper left",
              framealpha=0.88, handletextpad=0.4, borderpad=0.5,
              title="Binding label", title_fontsize=7.5)
    plt.tight_layout()
    save_fig(out_dir / "fig4A_scatter")

    # ── Panel B — PCA strip plot ──────────────────────────────────────────────
    print("\nPanel B — PCA strip")
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x_lo = df_plot["pc1"].min() - 0.4
    x_hi = df_plot["pc1"].max() + 0.4

    ax.axvspan(x_lo,       ca_pc1_max, color=GRP_COL["Ca-like"],   alpha=0.08)
    ax.axvspan(ca_pc1_max, la_pc1_min, color=GRP_COL["Transition"], alpha=0.08)
    ax.axvspan(la_pc1_min, x_hi,       color=GRP_COL["La-like"],   alpha=0.08)
    ax.axvline(ca_pc1_max, color=GRP_COL["Ca-like"],  lw=1.0, ls="--", alpha=0.7)
    ax.axvline(la_pc1_min, color=GRP_COL["La-like"],  lw=1.0, ls="--", alpha=0.7)

    y_positions = {lbl: i for i, lbl in enumerate(reversed(LABEL_ORDER))}
    for lbl in LABEL_ORDER:
        sub = df_plot[df_plot["binding_label"] == lbl]
        if sub.empty: continue
        ms = MSTYLE[lbl]
        jitter = rng.uniform(-0.28, 0.28, size=len(sub))
        ax.scatter(sub["pc1"], y_positions[lbl] + jitter,
                   marker=ms["marker"], s=ms["s"] * 0.85, linewidths=ms["lw"],
                   c=LCOL[lbl], edgecolors="white", alpha=0.88, zorder=ms["zorder"])

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(
        [f"{lbl}  (n={len(df_plot[df_plot['binding_label']==lbl])})"
         for lbl in reversed(LABEL_ORDER)], fontsize=8.5)
    ax.set_xlabel(f"PC1 score  ({ev[0]*100:.1f}% variance — La-binding strength →)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linewidth=0.3, alpha=0.35)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.6, len(LABEL_ORDER) - 0.4)

    ymax = len(LABEL_ORDER) - 0.42
    for grp, x0, x1 in [("Ca-like",    x_lo,       ca_pc1_max),
                         ("Transition", ca_pc1_max, la_pc1_min),
                         ("La-like",    la_pc1_min, x_hi)]:
        n = (df_plot["pca_group"] == grp).sum()
        ax.text((x0 + x1) / 2, ymax, f"{grp}\n(n={n})",
                ha="center", va="top", fontsize=8,
                color=GRP_COL[grp], fontweight="bold")

    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    save_fig(out_dir / "fig4_pca_strip")

    print("\nDone.")


if __name__ == "__main__":
    main()
