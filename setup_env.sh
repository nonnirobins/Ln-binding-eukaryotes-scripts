#!/usr/bin/env bash
# Build the analysis environment into ./env
#
#   bash setup_env.sh          # conda/mamba if available, else pip venv
#   bash setup_env.sh pip      # force the pip venv route
#
# Afterwards run the scripts with ./env/bin/python <script>.py -- no activation needed.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -e env ]; then
    echo "./env already exists. Remove it first to rebuild:  rm -rf env"
    exit 1
fi

route="${1:-auto}"

if [ "$route" != "pip" ] && command -v mamba >/dev/null 2>&1; then
    echo "==> Building conda environment with mamba"
    mamba env create -p ./env -f environment.yml
elif [ "$route" != "pip" ] && command -v conda >/dev/null 2>&1; then
    echo "==> Building conda environment with conda"
    conda env create -p ./env -f environment.yml
else
    echo "==> Building pip virtualenv (Python 3.11+ required)"
    python3 -m venv env
    ./env/bin/pip install --upgrade pip
    ./env/bin/pip install -r requirements.txt
fi

echo "==> Verifying"
./env/bin/python - <<'PY'
import numpy, scipy, pandas, matplotlib, seaborn, sklearn, Bio, gemmi, tmtools
from tmtools import tm_align
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
for m in (numpy, scipy, pandas, matplotlib, seaborn, sklearn, Bio, gemmi, tmtools):
    print(f"  {m.__name__:14s} {getattr(m, '__version__', 'n/a')}")
PY

echo
echo "Done. Example:"
echo "  ./env/bin/python Superimposition_pocket_characterization.py \\"
echo "      Crystal_references/Ln_xoxf_6oc6.pdb <query_dir> --print-hits"
