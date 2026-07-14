#!/usr/bin/env python3
"""
af3_stats.py
------------
Extract AlphaFold3 confidence statistics from PQQ-dependent MDH structure
predictions: metal–protein iPTM, coordinating residue pLDDT values, PQQ
pLDDT, and the coordinating oxygen inventory.

Statistics extracted per structure:
  metal_protein_iptm  — chain_pair_iptm[0][1] from *_summary_confidences.json;
                        measures how well the predicted metal position is
                        consistent with the protein conformation
  pqq_plddt           — mean B-factor (pLDDT) across all PQQ atoms (chain C)
  n_coord_oxygens     — number of protein/PQQ oxygens within --cutoff Å of the metal
  res1..res4          — up to four coordinating residues (by sequence order):
                          type (3-letter AA), sequence position, CA pLDDT
  o1..o9              — coordinating oxygens sorted by distance:
                          source label (e.g. ASP45-OD1, PQQ-O4A), distance in Å

AF3 CIF chain convention (protenix output):
  chain A — protein
  chain B — metal (La3+ or Ca2+)
  chain C — PQQ cofactor

Input directories may be either:
  • Flat: directory containing *.cif files directly; confidence JSONs expected
    in the same directory as matching *_summary_confidences.json
  • Job-dir: directory of subdirectories each containing *_model.cif and
    *_summary_confidences.json (the standard protenix / AF3 output layout)

La and Ca structures are matched by stripping any -pqq-la / -pqq-ca suffix,
and output as a single row per protein with la_ and ca_ prefixed columns.

Usage:
  python af3_stats.py --la-dir path/to/La-alone --ca-dir path/to/Ca-alone
  python af3_stats.py --la-dir path/to/La-alone   # La only
  python af3_stats.py --la-dir ... --ca-dir ... --cutoff 3.5 --out stats.tsv

Dependencies: gemmi
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import gemmi


COORD_CUTOFF = 3.5

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEP", "TPO",
}


# ── metal / PQQ extraction ────────────────────────────────────────────────────

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


def analyze_coordination(st, metal_pos, pqq_atoms, cutoff=COORD_CUTOFF):
    """
    Return (coord_residues, coord_oxygens, pqq_plddt).

    coord_residues — gemmi Residue objects for each unique AA residue that
                     contributes an oxygen within cutoff of the metal,
                     sorted by sequence position
    coord_oxygens  — list of (label, dist) sorted by distance; labels like
                     "ASP45-OD1" or "PQQ-O4A"
    pqq_plddt      — mean B-factor (pLDDT) across all PQQ atoms,
                     or None if no PQQ
    """
    if metal_pos is None:
        return [], [], None

    seen_seqids = set()
    coord_residues = []
    coord_oxygens = []

    for res in st[0]["A"]:
        if res.name not in AA3:
            continue
        seqid = int(str(res.seqid).strip())
        for atom in res:
            if atom.element == gemmi.Element("O"):
                d = atom.pos.dist(metal_pos)
                if d <= cutoff:
                    coord_oxygens.append((f"{res.name}{seqid}-{atom.name}", round(d, 3)))
                    if seqid not in seen_seqids:
                        seen_seqids.add(seqid)
                        coord_residues.append(res)

    coord_residues.sort(key=lambda r: int(str(r.seqid).strip()))

    pqq_biso = [atom.b_iso for atom in pqq_atoms]
    pqq_plddt = round(sum(pqq_biso) / len(pqq_biso), 1) if pqq_biso else None
    for atom in pqq_atoms:
        if atom.element == gemmi.Element("O"):
            d = atom.pos.dist(metal_pos)
            if d <= cutoff:
                coord_oxygens.append((f"PQQ-{atom.name}", round(d, 3)))

    coord_oxygens.sort(key=lambda x: x[1])
    return coord_residues, coord_oxygens, pqq_plddt


def load_iptm(json_path):
    """Read chain_pair_iptm[0][1] (metal–protein iPTM) from a confidences JSON."""
    try:
        d = json.loads(Path(json_path).read_text())
        mat = d.get("chain_pair_iptm", [])
        if mat and len(mat[0]) > 1:
            v = mat[0][1]
            return round(v, 3) if v is not None else None
    except Exception:
        pass
    return None


def load_pqq_protein_iptm(json_path):
    """Read chain_pair_iptm[0][2] (protein–PQQ iPTM) from a confidences JSON."""
    try:
        d = json.loads(Path(json_path).read_text())
        mat = d.get("chain_pair_iptm", [])
        if mat and len(mat[0]) > 2:
            v = mat[0][2]
            return round(v, 3) if v is not None else None
    except Exception:
        pass
    return None


# ── file collection ───────────────────────────────────────────────────────────

def collect_jobs(directory):
    """Collect (cif_path, json_path) pairs from flat or job-dir layout.

    Returns {normalised_key: (cif_path, json_path|None)}.
    """
    d = Path(directory)
    result = {}
    subdirs = [s for s in sorted(d.iterdir()) if s.is_dir()]
    if subdirs:
        for subdir in subdirs:
            cifs = sorted(subdir.glob("*_model.cif"))
            jsons = sorted(subdir.glob("*_summary_confidences.json"))
            if cifs:
                result[_norm(subdir.name)] = (cifs[0], jsons[0] if jsons else None)
    else:
        for cif in sorted(d.glob("*.cif")):
            stem = cif.stem
            json_candidates = list(d.glob(f"{stem}*_summary_confidences.json"))
            if not json_candidates:
                # Try same stem, replacing _model with _summary_confidences
                json_candidates = list(d.glob(
                    stem.replace("_model", "") + "*_summary_confidences.json"))
            result[_norm(stem)] = (cif, json_candidates[0] if json_candidates else None)
    return result


def _norm(name):
    return re.sub(r"[-_]pqq[-_](la|ca)$", "", name, flags=re.IGNORECASE)


# ── output schema ─────────────────────────────────────────────────────────────

def _side_columns(prefix):
    cols = [
        f"{prefix}n_coord_oxygens",
        f"{prefix}metal_protein_iptm",
        f"{prefix}pqq_protein_iptm",
        f"{prefix}pqq_plddt",
        f"{prefix}coord_residues",   # semicolon-delimited: TYPE:POS:PLDDT for all residues
    ]
    for j in range(1, 10):
        cols += [f"{prefix}o{j}_source", f"{prefix}o{j}_dist"]
    return cols


COLUMNS = ["structure"] + _side_columns("la_") + _side_columns("ca_")


def extract_stats(cif_path, json_path, cutoff):
    """Return a dict of stat values for one structure (no prefix)."""
    st = gemmi.read_structure(str(cif_path))
    metal_pos, pqq_atoms = find_metal_and_pqq(st)
    coord_res, coord_oxy, pqq_plddt = analyze_coordination(
        st, metal_pos, pqq_atoms, cutoff)

    res_parts = []
    for res in coord_res:
        seqid = int(str(res.seqid).strip())
        ca = res.find_atom("CA", "\0")
        plddt = round(ca.b_iso, 1) if ca else "?"
        res_parts.append(f"{res.name}:{seqid}:{plddt}")

    out = {
        "n_coord_oxygens":  len(coord_oxy),
        "metal_protein_iptm": load_iptm(json_path) if json_path else "",
        "pqq_protein_iptm": load_pqq_protein_iptm(json_path) if json_path else "",
        "pqq_plddt":        pqq_plddt if pqq_plddt is not None else "",
        "coord_residues":   ";".join(res_parts),
    }
    for j, (label, dist) in enumerate(coord_oxy[:9], 1):
        out[f"o{j}_source"] = label
        out[f"o{j}_dist"]   = dist
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Extract AF3 confidence statistics from PQQ-MDH structure predictions: "
            "metal–protein iPTM, coordinating residue pLDDT values, PQQ pLDDT, "
            "and coordinating oxygen inventory."
        )
    )
    ap.add_argument("--la-dir", required=True,
                    help="Directory of La-bound AF3 job outputs")
    ap.add_argument("--ca-dir", default=None,
                    help="Directory of Ca-bound AF3 job outputs (optional)")
    ap.add_argument("--cutoff", type=float, default=COORD_CUTOFF,
                    help=f"Metal–oxygen cutoff in Å (default: {COORD_CUTOFF})")
    ap.add_argument("--out", default="af3_stats.tsv",
                    help="Output TSV path (default: af3_stats.tsv)")
    args = ap.parse_args()

    la_jobs = collect_jobs(args.la_dir)
    ca_jobs = collect_jobs(args.ca_dir) if args.ca_dir else {}
    print(f"Found {len(la_jobs)} La structures, {len(ca_jobs)} Ca structures",
          file=sys.stderr)

    all_keys = sorted(la_jobs.keys() | ca_jobs.keys())

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()

        for key in all_keys:
            row = {c: "" for c in COLUMNS}
            row["structure"] = key

            if key in la_jobs:
                cif, jsn = la_jobs[key]
                try:
                    stats = extract_stats(cif, jsn, args.cutoff)
                    for k, v in stats.items():
                        row[f"la_{k}"] = v
                    n = stats["n_coord_oxygens"]
                    iptm = stats["metal_protein_iptm"]
                    pqq = stats["pqq_plddt"]
                    print(f"  {key:40s}  La  n_oxy={n:<3}  iptm={iptm!s:<6}  pqq_plddt={pqq}",
                          file=sys.stderr)
                except Exception as e:
                    print(f"  WARN {key} La: {e}", file=sys.stderr)

            if key in ca_jobs:
                cif, jsn = ca_jobs[key]
                try:
                    stats = extract_stats(cif, jsn, args.cutoff)
                    for k, v in stats.items():
                        row[f"ca_{k}"] = v
                    if key not in la_jobs:
                        n = stats["n_coord_oxygens"]
                        iptm = stats["metal_protein_iptm"]
                        pqq = stats["pqq_plddt"]
                        print(f"  {key:40s}  Ca  n_oxy={n:<3}  iptm={iptm!s:<6}  pqq_plddt={pqq}",
                              file=sys.stderr)
                except Exception as e:
                    print(f"  WARN {key} Ca: {e}", file=sys.stderr)

            writer.writerow(row)

    print(f"\nWrote {len(all_keys)} rows → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
