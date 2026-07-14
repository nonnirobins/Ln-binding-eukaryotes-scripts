#!/usr/bin/env python3
"""
La_pocket_characterization.py
------------------------------
Profiles the La3+-coordinating pocket of PQQ-dependent methanol dehydrogenase
(MDH) structures by superimposing each query onto a La3+-containing reference
structure and counting ASP/GLU/ASN sidechain oxygens within a distance cutoff
of the reference La3+ position.

Workflow:
  1. Load a reference structure (PDB or mmCIF) that contains a La3+ atom.
  2. For each query structure file (.pdb/.ent/.cif/.mmcif) in a given directory:
       a. Extract Cα traces and perform TM-align (via tmtools) to the reference.
       b. Apply the inverse rigid transform to move the query into the reference frame.
       c. Find ASP/GLU/ASN sidechain oxygens within --cutoff Å of the reference La3+.
       d. Build a pocket profile string (e.g. "2D_1E_1N") encoding residue counts.
  3. Write results to a TSV (structure, profile, hit residues) and copy each
     query file into a subdirectory named by its profile for easy browsing.

Usage:
  python La_pocket_characterization.py <ref_structure> <query_dir> [options]

  Positional:
    ref_structure   Reference PDB/ENT/mmCIF file containing La3+
    query_dir       Directory of query structure files to profile

  Key options:
    --cutoff FLOAT      Oxygen–La distance cutoff in Å (default: 5.0)
    --chain-ref ID      Chain to use from the reference for TM-align (default: all)
    --chain-query ID    Chain to use from each query (default: all)
    --outdir DIR        Base directory for per-profile subdirs (default: Pocket_profiles)
    --tsv FILE          Output TSV filename (default: pocket_profiles.tsv)
    --write-aligned DIR Save TM-aligned query PDBs to DIR for visualisation
    --print-hits        Print per-structure profile and hit residues to stdout

Dependencies: biopython, tmtools, numpy
"""

import os
import sys
import argparse
import shutil
import numpy as np

from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from tmtools import tm_align  # returns res.u (rotation) and res.t (translation)


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

# Residues considered (sidechain oxygen-mediated)
PROFILE_CODE = {"ASP": "D", "GLU": "E", "ASN": "N"}
OXYGEN_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "ASN": ["OD1"],
}


def get_parser_for_path(path: str):
    p = path.lower()
    if p.endswith(".cif") or p.endswith(".mmcif"):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def load_structure(path: str, struct_id: str):
    parser = get_parser_for_path(path)
    return parser.get_structure(struct_id, path)


def find_lanthanum_atoms(structure):
    """
    mmCIF element fields can be flaky; name/resname matching is robust.
    """
    la_atoms = []
    for atom in structure.get_atoms():
        name = atom.get_name().strip().upper()
        res = atom.get_parent()
        resname = res.get_resname().strip().upper() if res is not None else ""
        element = getattr(atom, "element", "")
        element = element.upper() if isinstance(element, str) else ""
        if name == "LA" or element == "LA" or resname in ("LA", "LAN"):
            la_atoms.append(atom)
    return la_atoms


def extract_ca_trace_and_seq(structure, chain_id=None):
    coords = []
    seq = []
    model = structure[0]
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain:
            # skip hetero/water
            if residue.id[0].strip() != "":
                continue
            if "CA" not in residue:
                continue
            resname = residue.get_resname().strip().upper()
            aa = AA3_TO_1.get(resname, "X")
            coords.append(residue["CA"].coord.astype(float))
            seq.append(aa)
    if not coords:
        return np.zeros((0, 3), dtype=float), ""
    return np.vstack(coords), "".join(seq)


def apply_inverse_rigid_transform(structure, R, t):
    """
    tmtools returns R,t such that:
        ref_aligned = R @ ref + t
    To move QUERY into REF frame:
        query_in_ref = R^T @ (query - t)
    """
    Rt = R.T
    for atom in structure.get_atoms():
        c = atom.coord.astype(float)
        atom.coord = Rt @ (c - t)


def find_oxygen_hits(structure, la_coord, cutoff=5.0, chain_id=None):
    """
    Keep the closest oxygen per residue among ASP/GLU/ASN.
    Returns dict[(chain,resseq,icode)] -> (resname, code, atomname, dist)
    """
    model = structure[0]
    best_by_residue = {}

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue

        for residue in chain:
            if residue.id[0].strip() != "":
                continue

            resname = residue.get_resname().upper()
            if resname not in OXYGEN_ATOMS:
                continue

            code = PROFILE_CODE[resname]
            rid = (chain.id, residue.id[1], residue.id[2].strip())

            for oname in OXYGEN_ATOMS[resname]:
                if oname not in residue:
                    continue
                d = float(np.linalg.norm(residue[oname].coord.astype(float) - la_coord))
                if d <= cutoff:
                    prev = best_by_residue.get(rid)
                    if prev is None or d < prev[3]:
                        best_by_residue[rid] = (resname, code, oname, d)

    return best_by_residue


def build_profile_string(best_by_residue):
    # profile like 2D_1E_1N (alphabetical)
    counts = {}
    for (_rid, (_resname, code, _oname, _d)) in best_by_residue.items():
        counts[code] = counts.get(code, 0) + 1
    if not counts:
        return "NO_HITS"
    parts = [f"{counts[c]}{c}" for c in sorted(counts.keys())]
    return "_".join(parts)


def format_hit_residues(best_by_residue):
    items = []
    for (chain, resseq, icode), (resname, _code, atom, d) in best_by_residue.items():
        icode_str = f"{icode}" if icode else ""
        items.append((d, f"{resname} {chain}:{resseq}{icode_str} {atom}({d:.2f})"))
    items.sort(key=lambda x: x[0])
    return "; ".join(s for _d, s in items)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Align queries to a La reference (tmtools TM-align), then profile the La pocket by "
            "ASP/GLU/ASN sidechain oxygen contacts within cutoff. "
            "Works with PDB/ENT/mmCIF inputs (reference can be PDB, queries can be mmCIF)."
        )
    )
    ap.add_argument("ref_structure", help="Reference structure (PDB/ENT/mmCIF) containing La3+")
    ap.add_argument("query_dir", help="Directory of query structures (.pdb/.ent/.cif/.mmcif)")

    ap.add_argument("--chain-ref", default=None, help="Chain used from reference for alignment (default: all)")
    ap.add_argument("--chain-query", default=None, help="Chain used from query for alignment+profiling (default: all)")
    ap.add_argument("--cutoff", type=float, default=5.0, help="Oxygen–La cutoff Å (default: 5.0)")

    ap.add_argument("--outdir", default="Pocket_profiles", help="Base output directory (default: Pocket_profiles)")
    ap.add_argument("--tsv", default="pocket_profiles.tsv", help="Output TSV (default: pocket_profiles.tsv)")
    ap.add_argument("--print-hits", action="store_true", help="Print per-structure profile and hits")
    ap.add_argument("--write-aligned", default=None, help="Optional dir to save aligned query PDBs")

    args = ap.parse_args()

    if not os.path.isfile(args.ref_structure):
        print(f"Reference not found: {args.ref_structure}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.query_dir):
        print(f"Query dir not found: {args.query_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    if args.write_aligned:
        os.makedirs(args.write_aligned, exist_ok=True)

    # Load reference (PDB ok)
    ref = load_structure(args.ref_structure, "ref")
    la_atoms = find_lanthanum_atoms(ref)
    if not la_atoms:
        print("No La found in reference.", file=sys.stderr)
        sys.exit(1)
    if len(la_atoms) > 1:
        print(f"Warning: found {len(la_atoms)} La atoms; using the first.", file=sys.stderr)
    la_coord = la_atoms[0].coord.astype(float).copy()

    ref_ca, ref_seq = extract_ca_trace_and_seq(ref, chain_id=args.chain_ref)
    if len(ref_seq) == 0:
        print("No CA atoms found in reference for alignment.", file=sys.stderr)
        sys.exit(1)

    # Accept queries in either format
    exts = (".pdb", ".ent", ".cif", ".mmcif")
    query_files = sorted(f for f in os.listdir(args.query_dir) if f.lower().endswith(exts))
    if not query_files:
        print(f"No query files with extensions {exts} found.", file=sys.stderr)
        sys.exit(1)

    io = PDBIO()

    with open(args.tsv, "w") as out:
        out.write("structure\tprofile\thit_residues\n")

        for fname in query_files:
            qpath = os.path.join(args.query_dir, fname)

            try:
                query = load_structure(qpath, "query")
            except Exception as e:
                print(f"[SKIP] parse failed {fname}: {e}", file=sys.stderr)
                continue

            qry_ca, qry_seq = extract_ca_trace_and_seq(query, chain_id=args.chain_query)
            if len(qry_seq) == 0:
                print(f"[SKIP] no CA atoms found for query selection: {fname}", file=sys.stderr)
                continue

            try:
                res = tm_align(ref_ca, qry_ca, ref_seq, qry_seq)
                R = np.array(res.u, dtype=float)
                t = np.array(res.t, dtype=float)
            except Exception as e:
                print(f"[SKIP] tmtools TM-align failed {fname}: {e}", file=sys.stderr)
                continue

            apply_inverse_rigid_transform(query, R, t)

            best_by_res = find_oxygen_hits(query, la_coord, cutoff=args.cutoff, chain_id=args.chain_query)
            profile = build_profile_string(best_by_res)
            hit_str = format_hit_residues(best_by_res) if best_by_res else ""

            dest_dir = os.path.join(args.outdir, profile)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(qpath, os.path.join(dest_dir, fname))

            out.write(f"{fname}\t{profile}\t{hit_str}\n")

            if args.print_hits:
                print(f"[{profile}] {fname}")
                if hit_str:
                    print(f"    {hit_str}")
                else:
                    print("    (no ASP/GLU/ASN oxygen hits within cutoff)")

            if args.write_aligned:
                # Write aligned query as PDB for easy visualization
                out_pdb = os.path.join(args.write_aligned, os.path.splitext(fname)[0] + ".pdb")
                io.set_structure(query)
                io.save(out_pdb)


if __name__ == "__main__":
    main()
