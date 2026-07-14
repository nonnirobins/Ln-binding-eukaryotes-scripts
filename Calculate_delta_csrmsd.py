#!/usr/bin/env python3
"""
compute_csrmsd.py
-----------------
Compute coordination-sphere RMSD (csRMSD) for PQQ-dependent MDH structures
against La3+ (XoxF) and Ca2+ (MxaF) crystal structure references, and report
ΔcsRMSD = ca_csrmsd − la_csrmsd.

csRMSD algorithm (ICP: Hungarian + Kabsch):
  1. Collect all oxygens within --cutoff Å of the metal from both the query
     AF3 model and the reference crystal structure, centred on their metal.
  2. Iteratively find the optimal one-to-one oxygen assignment (Hungarian) and
     the optimal rotation (Kabsch) until the assignment converges.
  3. Unmatched oxygens are penalised by COORD_CUTOFF² each.
  4. csRMSD = sqrt((matched_sq + n_unmatched × cutoff²) / max(n_ref, n_query))

ΔcsRMSD = ca_csrmsd_vs_MxaF − la_csrmsd_vs_XoxF
  Positive → more XoxF-like (La-binding preference)
  Negative → more MxaF-like (Ca-binding preference)

Input directories may be either:
  • Flat: directory containing *.cif files directly
  • Job-dir: directory of subdirectories each containing a *_model.cif
    (the standard protenix / AF3 output layout)

La and Ca structures are matched by stripping any -pqq-la / -pqq-ca suffix
from the subdirectory or file name.

Usage:
  python compute_csrmsd.py \\
      --la-ref  xoxf.pdb \\
      --ca-ref  mxaf_pdb1h4i.pdb \\
      --la-dir  path/to/La-alone \\
      --ca-dir  path/to/Ca-alone \\
      --out     csrmsd.tsv

  If --ca-dir is omitted, only la_csrmsd is reported (ca_csrmsd and
  delta_csrmsd will be empty).

Dependencies: gemmi, numpy, scipy
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import gemmi
import numpy as np
from scipy.optimize import linear_sum_assignment


COORD_CUTOFF = 3.5

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEP", "TPO",
}


# ── coordination sphere RMSD ──────────────────────────────────────────────────

def kabsch_rmsd(P, Q):
    P, Q = np.array(P, dtype=float), np.array(Q, dtype=float)
    p0, q0 = P.mean(0), Q.mean(0)
    Pc, Qc = P - p0, Q - q0
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum() / len(P)))


def coord_sphere_rmsd(query_oxygens, ref_oxygens, cutoff=COORD_CUTOFF, max_iter=30):
    """
    Metal-centred coordination sphere RMSD via ICP (Hungarian + Kabsch).

    Returns (csrmsd, n_ref, n_query) or (None, 0, 0) when ref is empty.
    Unmatched oxygens are penalised by cutoff² so the metric encodes both
    geometry and coordination number. Denominator = max(n_ref, n_query).
    """
    if not ref_oxygens:
        return None, 0, 0

    R = np.array(ref_oxygens, dtype=float)
    n_ref = len(R)

    if not query_oxygens:
        return round(float(cutoff), 3), n_ref, 0

    Q = np.array(query_oxygens, dtype=float)
    n_q = len(Q)

    rot = np.eye(3)
    prev_col_ind = None

    for _ in range(max_iter):
        Q_rot = Q @ rot.T
        diff = Q_rot[:, np.newaxis, :] - R[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        if prev_col_ind is not None and np.array_equal(col_ind, prev_col_ind):
            break
        prev_col_ind = col_ind.copy()
        Q_m, R_m = Q_rot[row_ind], R[col_ind]
        H = Q_m.T @ R_m
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        rot = (Vt.T @ np.diag([1, 1, d]) @ U.T) @ rot

    Q_rot = Q @ rot.T
    diff = Q_rot[:, np.newaxis, :] - R[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matched_sq = (dist_matrix[row_ind, col_ind] ** 2).sum()
    n_unmatched = abs(n_ref - n_q)
    denom = max(n_ref, n_q)
    csrmsd = float(np.sqrt((matched_sq + n_unmatched * cutoff ** 2) / denom))
    return round(csrmsd, 3), n_ref, n_q


# ── oxygen extraction ─────────────────────────────────────────────────────────

def get_pdb_coord_oxygens_centered(pdb_path, cutoff=COORD_CUTOFF):
    """Metal-centred coordinating oxygen positions from a reference PDB/CIF."""
    st = gemmi.read_structure(str(pdb_path))
    metal_pos = None
    non_organic = {
        gemmi.Element("C"), gemmi.Element("N"), gemmi.Element("O"),
        gemmi.Element("S"), gemmi.Element("H"), gemmi.Element("P"),
        gemmi.Element("X"),
    }
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.element not in non_organic:
                    metal_pos = atom.pos
                    break
            if metal_pos:
                break
        if metal_pos:
            break
    if metal_pos is None:
        return []
    return [
        [atom.pos.x - metal_pos.x,
         atom.pos.y - metal_pos.y,
         atom.pos.z - metal_pos.z]
        for chain in st[0]
        for res in chain
        for atom in res
        if atom.element == gemmi.Element("O") and atom.pos.dist(metal_pos) <= cutoff
    ]


def find_metal_and_pqq(st):
    """Find metal position (chain B) and PQQ atoms (chain C) in an AF3 CIF."""
    metal_pos, pqq_atoms = None, []
    model = st[0]
    chain_names = [ch.name for ch in model]
    if "B" in chain_names:
        for res in model["B"]:
            for atom in res:
                metal_pos = atom.pos
                break
            break
    if "C" in chain_names:
        for res in model["C"]:
            pqq_atoms = list(res)
            break
    return metal_pos, pqq_atoms


def get_af3_coord_oxygens_centered(st, metal_pos, pqq_atoms, cutoff=COORD_CUTOFF):
    """Metal-centred coordinating oxygen positions from an AF3 CIF model."""
    if metal_pos is None:
        return []
    positions = []
    for res in st[0]["A"]:
        if res.name not in AA3:
            continue
        for atom in res:
            if atom.element == gemmi.Element("O") and atom.pos.dist(metal_pos) <= cutoff:
                positions.append([
                    atom.pos.x - metal_pos.x,
                    atom.pos.y - metal_pos.y,
                    atom.pos.z - metal_pos.z,
                ])
    for atom in pqq_atoms:
        if atom.element == gemmi.Element("O") and atom.pos.dist(metal_pos) <= cutoff:
            positions.append([
                atom.pos.x - metal_pos.x,
                atom.pos.y - metal_pos.y,
                atom.pos.z - metal_pos.z,
            ])
    return positions


# ── file collection ───────────────────────────────────────────────────────────

def collect_cifs(directory):
    """Collect CIF files from a flat directory or a job-dir tree.

    Returns {normalised_key: path}.
    Job-dir layout: each subdir contains a *_model.cif (protenix / AF3 output).
    Flat layout: *.cif files sit directly in the directory.
    Key normalisation strips -pqq-la / -pqq-ca suffixes so La and Ca
    structures for the same protein share a common key.
    """
    d = Path(directory)
    result = {}
    subdirs = [s for s in sorted(d.iterdir()) if s.is_dir()]
    if subdirs:
        for subdir in subdirs:
            files = sorted(subdir.glob("*_model.cif"))
            if not files:
                files = sorted(subdir.glob("*.cif"))
            if files:
                key = _normalise(subdir.name)
                result[key] = files[0]
    else:
        for f in sorted(d.glob("*.cif")):
            result[_normalise(f.stem)] = f
    return result


def _normalise(name):
    return re.sub(r"[-_]pqq[-_](la|ca)$", "", name, flags=re.IGNORECASE)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute csRMSD of AF3 MDH structures vs. La3+ (XoxF) and Ca2+ (MxaF) "
            "references, then report ΔcsRMSD = ca_csrmsd − la_csrmsd."
        )
    )
    ap.add_argument("--la-ref", required=True,
                    help="Reference La-bound structure (XoxF PDB/CIF)")
    ap.add_argument("--ca-ref", required=True,
                    help="Reference Ca-bound structure (MxaF PDB/CIF)")
    ap.add_argument("--la-dir", required=True,
                    help="Directory of La-bound AF3 structures")
    ap.add_argument("--ca-dir", default=None,
                    help="Directory of Ca-bound AF3 structures (optional)")
    ap.add_argument("--cutoff", type=float, default=COORD_CUTOFF,
                    help=f"Metal–oxygen cutoff in Å (default: {COORD_CUTOFF})")
    ap.add_argument("--out", default="csrmsd.tsv",
                    help="Output TSV path (default: csrmsd.tsv)")
    args = ap.parse_args()

    # Load reference oxygens once
    la_ref_oxy = get_pdb_coord_oxygens_centered(args.la_ref, args.cutoff)
    ca_ref_oxy = get_pdb_coord_oxygens_centered(args.ca_ref, args.cutoff)
    print(f"XoxF reference: {len(la_ref_oxy)} coordinating oxygens", file=sys.stderr)
    print(f"MxaF reference: {len(ca_ref_oxy)} coordinating oxygens", file=sys.stderr)

    la_cifs = collect_cifs(args.la_dir)
    ca_cifs = collect_cifs(args.ca_dir) if args.ca_dir else {}
    print(f"Found {len(la_cifs)} La structures, {len(ca_cifs)} Ca structures",
          file=sys.stderr)

    all_keys = sorted(la_cifs.keys() | ca_cifs.keys())

    columns = [
        "structure",
        "la_csrmsd_vs_xoxf", "la_n_ref_oxy", "la_n_query_oxy",
        "ca_csrmsd_vs_mxaf", "ca_n_ref_oxy", "ca_n_query_oxy",
        "delta_csrmsd",
    ]

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()

        for key in all_keys:
            row = {c: "" for c in columns}
            row["structure"] = key

            # La csRMSD
            if key in la_cifs:
                try:
                    st = gemmi.read_structure(str(la_cifs[key]))
                    metal_pos, pqq_atoms = find_metal_and_pqq(st)
                    q_oxy = get_af3_coord_oxygens_centered(
                        st, metal_pos, pqq_atoms, args.cutoff)
                    csrmsd, n_ref, n_q = coord_sphere_rmsd(
                        q_oxy, la_ref_oxy, args.cutoff)
                    row["la_csrmsd_vs_xoxf"] = csrmsd if csrmsd is not None else ""
                    row["la_n_ref_oxy"]  = n_ref
                    row["la_n_query_oxy"] = n_q
                except Exception as e:
                    print(f"  WARN {key} La: {e}", file=sys.stderr)

            # Ca csRMSD
            if key in ca_cifs:
                try:
                    st = gemmi.read_structure(str(ca_cifs[key]))
                    metal_pos, pqq_atoms = find_metal_and_pqq(st)
                    q_oxy = get_af3_coord_oxygens_centered(
                        st, metal_pos, pqq_atoms, args.cutoff)
                    csrmsd, n_ref, n_q = coord_sphere_rmsd(
                        q_oxy, ca_ref_oxy, args.cutoff)
                    row["ca_csrmsd_vs_mxaf"] = csrmsd if csrmsd is not None else ""
                    row["ca_n_ref_oxy"]  = n_ref
                    row["ca_n_query_oxy"] = n_q
                except Exception as e:
                    print(f"  WARN {key} Ca: {e}", file=sys.stderr)

            # ΔcsRMSD
            if row["la_csrmsd_vs_xoxf"] != "" and row["ca_csrmsd_vs_mxaf"] != "":
                row["delta_csrmsd"] = round(
                    float(row["ca_csrmsd_vs_mxaf"]) - float(row["la_csrmsd_vs_xoxf"]), 3)

            writer.writerow(row)
            print(
                f"  {key:40s}  La={row['la_csrmsd_vs_xoxf']:>6}  "
                f"Ca={row['ca_csrmsd_vs_mxaf']:>6}  Δ={row['delta_csrmsd']:>7}",
                file=sys.stderr,
            )

    print(f"\nWrote {len(all_keys)} rows → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
