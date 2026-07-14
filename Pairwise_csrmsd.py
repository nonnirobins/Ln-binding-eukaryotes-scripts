#!/usr/bin/env python3
"""
Pairwise_csrmsd.py
------------------
Generate the pairwise coordination-sphere RMSD heatmap (Figure 4 Panel C):
a seaborn clustermap of the full pairwise csRMSD matrix with hierarchical
clustering (average linkage, k=2) and three row/column annotation tracks:

  Binding label  — colour-coded by binding_label from the assessment TSV
  Cluster        — k=2 silhouette cluster (La cohort / Ca cohort)
  n coord. O     — number of coordinating oxygens (viridis scale)

Verified La Binding and Verified Ca Binding proteins are additionally marked
with star / diamond overlays on the annotation strips.

Cluster 1 is always assigned as La cohort (contains more Verified La Binding
proteins); cluster 2 becomes Ca cohort.

Requires pca_group to be present in the assessment TSV — run
PCA_classification.py first if it is missing.

Usage:
  python Pairwise_csrmsd.py
  python Pairwise_csrmsd.py --assess /path/to/la_binding_assessment.tsv \\
                             --matrix /path/to/pairwise_csrmsd_matrix.tsv
  python Pairwise_csrmsd.py --out-dir /path/to/output/

Dependencies: numpy, pandas, matplotlib, scipy, scikit-learn, seaborn
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import seaborn as sns
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────
_BASE   = Path("/groups/banfield/projects/multienv/corkscrew/supplementary_structures")
_ASSESS = _BASE / "la_binding_assessment.tsv"
_MATRIX = _BASE / "pairwise_csrmsd_matrix.tsv"

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
PCA_COL  = {"La-like": "#e97132", "Transition": "#888888", "Ca-like": "#007d7d"}
NO_PCA   = "#e0e0e0"
CLUST_PAL = {1: "#c8920a", 2: "#1a2f6b"}

VERIFIED_MARKER = {
    "Verified La Binding": dict(marker="*", s=55, color="#ffffff", zorder=10),
    "Verified Ca Binding": dict(marker="D", s=28, color="#ffffff", zorder=10),
}


def main():
    ap = argparse.ArgumentParser(
        description="Pairwise csRMSD heatmap (k=2 clustering) for PQQ-MDH assessment."
    )
    ap.add_argument("--assess",  default=str(_ASSESS),
                    help="Assessment TSV (default: la_binding_assessment.tsv)")
    ap.add_argument("--matrix",  default=str(_MATRIX),
                    help="Pairwise csRMSD matrix TSV (default: pairwise_csrmsd_matrix.tsv)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for figures (default: same dir as --assess)")
    args = ap.parse_args()

    assess_path = Path(args.assess)
    matrix_path = Path(args.matrix)
    out_dir     = Path(args.out_dir) if args.out_dir else assess_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load assessment ───────────────────────────────────────────────────────
    df_all  = pd.read_csv(assess_path, sep="\t")
    df_kept = df_all[df_all["quality_control"] == "kept"].copy()

    # ── load & filter matrix ──────────────────────────────────────────────────
    mat_full  = pd.read_csv(matrix_path, sep="\t", index_col=0)
    kept_accs = set(df_kept["accession"].values) if "accession" in df_kept.columns \
                else set(df_kept.index)

    def _base(key):
        return re.sub(r'_(domain|contig).*$', '', key)

    keep_mask  = [_base(a) in kept_accs for a in mat_full.index]
    mat_df     = mat_full.loc[keep_mask, keep_mask]
    accessions = list(mat_df.index)
    mat        = mat_df.values.astype(float)
    print(f"Matrix: {len(accessions)} × {len(accessions)}")

    # ── metadata lookup ───────────────────────────────────────────────────────
    df_all["_domain_key"] = df_all.apply(
        lambda r: r["accession"] if pd.isna(r.get("domain"))
                  else f"{r['accession']}_{r['domain']}",
        axis=1,
    )
    df_meta = df_all.set_index("_domain_key")

    def get_val(key, col, default=None):
        for k in [re.sub(r'_contig.*$', '', key), key]:
            if k in df_meta.index:
                v = df_meta.loc[k, col]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                return v if pd.notna(v) else default
        return default

    labels_hm  = [get_val(a, "binding_label", "Unknown") for a in accessions]
    n_oxy_vals = [get_val(a, "la_n_coord_oxygens")        for a in accessions]

    # ── oxygen colour scale ───────────────────────────────────────────────────
    oxy_finite = [int(v) for v in n_oxy_vals if v is not None]
    oxy_min, oxy_max = min(oxy_finite), max(oxy_finite)
    oxy_norm = mcolors.Normalize(vmin=oxy_min, vmax=oxy_max)
    oxy_cmap = matplotlib.colormaps["viridis"]
    NO_OXY   = "#e0e0e0"

    def oxy_to_hex(v):
        return NO_OXY if v is None else mcolors.to_hex(oxy_cmap(oxy_norm(int(v))))

    # ── hierarchical clustering k=2 ───────────────────────────────────────────
    condensed   = squareform(mat, checks=False)
    row_linkage = linkage(condensed, method="average")

    raw = fcluster(row_linkage, t=2, criterion="maxclust")
    la_counts = {ci: sum(1 for i, ci2 in enumerate(raw)
                         if ci2 == ci and labels_hm[i] == "Verified La Binding")
                 for ci in [1, 2]}
    if la_counts[1] < la_counts[2]:
        raw = 3 - raw

    sil = silhouette_score(mat, raw, metric="precomputed")
    print(f"Silhouette k=2: {sil:.4f}")

    clust_names = {1: "La cohort", 2: "Ca cohort"}

    # ── row/col colour annotations ────────────────────────────────────────────
    row_colors = pd.DataFrame({
        "Binding label": [LCOL.get(l, "#ffffff") for l in labels_hm],
        "Cluster":       [CLUST_PAL[c]           for c in raw],
        "n coord. O":    [oxy_to_hex(v)           for v in n_oxy_vals],
    }, index=accessions)

    # ── clustermap ────────────────────────────────────────────────────────────
    print("\nPanel C — heatmap")
    plt.rcParams.update({'font.size': 12})
    g = sns.clustermap(
        mat_df,
        row_linkage=row_linkage, col_linkage=row_linkage,
        row_colors=row_colors,   col_colors=row_colors,
        cmap="Blues", vmin=0, vmax=mat[mat < 3.4].max(),
        xticklabels=False, yticklabels=False, linewidths=0,
        figsize=(18, 18), dendrogram_ratio=0.12, colors_ratio=0.036,
        cbar_pos=(0.02, 0.82, 0.03, 0.14), tree_kws={"linewidths": 1.5},
    )
    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        for line in ax.lines:
            line.set_linewidth(1.5)

    # verified protein markers on annotation strips
    leaf_order = g.dendrogram_row.reordered_ind
    n_bands    = len(row_colors.columns)
    ax_rc = g.ax_row_colors
    ax_cc = g.ax_col_colors
    for j, orig_idx in enumerate(leaf_order):
        lbl = labels_hm[orig_idx]
        if lbl not in VERIFIED_MARKER: continue
        vm = VERIFIED_MARKER[lbl]
        ax_rc.scatter(0.5, j + 0.5, marker=vm["marker"], s=vm["s"],
                      c=vm["color"], linewidths=0.3, edgecolors="#888888",
                      zorder=vm["zorder"], clip_on=False)
        ax_cc.scatter(j + 0.5, n_bands - 0.5, marker=vm["marker"], s=vm["s"],
                      c=vm["color"], linewidths=0.3, edgecolors="#888888",
                      zorder=vm["zorder"], clip_on=False)

    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
    g.cax.set_ylabel("csRMSD (Å)", fontsize=12, rotation=270, labelpad=14)
    g.cax.tick_params(labelsize=11)
    g.fig.suptitle("Pairwise coordination sphere RMSD — average linkage, k=2",
                   y=1.005, fontsize=15)

    g.fig.legend(
        handles=[mpatches.Patch(facecolor=LCOL[l], edgecolor="#555555",
                                linewidth=0.5, label=l)
                 for l in LABEL_ORDER],
        title="Binding label", title_fontsize=12, fontsize=11,
        loc="lower left", bbox_to_anchor=(0.02, 0.02),
        framealpha=0.9, edgecolor="#cccccc")

    g.fig.legend(
        handles=[mpatches.Patch(facecolor=CLUST_PAL[ci], edgecolor="#555555",
                                linewidth=0.5,
                                label=f"{clust_names[ci]} (n={int((raw==ci).sum())})")
                 for ci in [1, 2]],
        title="Silhouette cluster (k=2)", title_fontsize=12, fontsize=11,
        loc="lower left", bbox_to_anchor=(0.58, 0.02),
        framealpha=0.9, edgecolor="#cccccc")

    g.fig.legend(
        handles=[mpatches.Patch(facecolor=oxy_to_hex(v), edgecolor="#555555",
                                linewidth=0.5, label=str(v))
                 for v in range(oxy_min, oxy_max + 1)]
              + [mpatches.Patch(facecolor=NO_OXY, edgecolor="#aaaaaa",
                                linewidth=0.5, label="No data")],
        title="n coord. oxygens", title_fontsize=12, fontsize=11,
        loc="lower left", bbox_to_anchor=(0.76, 0.02),
        framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    stem = out_dir / "pairwise_csrmsd_heatmap_k2"
    for ext in ("pdf", "png", "svg"):
        g.fig.savefig(f"{stem}.{ext}",
                      dpi=300 if ext == "pdf" else 200,
                      bbox_inches="tight", format=ext)
        print(f"  Saved {stem}.{ext}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
