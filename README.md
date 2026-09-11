# Ln-binding-eukaryotes-scripts
Code for running computationally assessing metal coordination site geometry of candidate lanthanide-binding eukaryotic proteins. 

## Setup

```bash
git clone git@github.com:nonnirobins/Ln-binding-eukaryotes-scripts.git
cd Ln-binding-eukaryotes-scripts
bash setup_env.sh
```

That builds a self-contained environment into `env/` (conda/mamba if available,
otherwise a pip virtualenv) and prints the installed versions. The `env/` directory
is deliberately not tracked by git — each user builds it locally.

Run the scripts with that interpreter directly; no activation needed:

```bash
./env/bin/python Superimposition_pocket_characterization.py \
    Crystal_references/Ln_xoxf_6oc6.pdb <query_dir> --print-hits
```

Conda users may instead activate it with `conda activate ./env`.

### Manual builds

| File | Route |
|---|---|
| `environment.yml` | conda/mamba, cross-platform: `mamba env create -p ./env -f environment.yml` |
| `requirements.txt` | pip/venv, Python 3.11+: `python3 -m venv env && ./env/bin/pip install -r requirements.txt` |
| `env.linux-64.lock.txt` | exact pinned conda builds, **linux-64 only**: `conda create -p ./env --file env.linux-64.lock.txt` then `./env/bin/pip install tmtools==0.3.0` |

Dependencies: Python 3.11, numpy, scipy, pandas, matplotlib, seaborn, scikit-learn,
biopython, gemmi, tmtools. No external binaries are required — TM-align is provided by
the `tmtools` Python bindings. Figures render headlessly via the matplotlib Agg backend,
so the scripts run fine over SSH and on compute nodes.

## Scripts

| Script | Purpose |
|---|---|
| `Superimposition_pocket_characterization.py` | TM-align queries onto a La3+ reference and profile ASP/GLU/ASN pocket oxygens |
| `Calculate_delta_csrmsd.py` | csRMSD vs. La (XoxF) and Ca (MxaF) references; reports ΔcsRMSD |
| `Retrieve_af3_stats.py` | Extract AF3 confidence stats (iPTM, pLDDT, coordinating oxygens) |
| `Pairwise_csrmsd.py` | Pairwise csRMSD heatmap with k=2 clustering |
| `PCA_classification.py` | PCA on (ΔcsRMSD, ΔΔE) and classification figure panels |

Each takes `--help`. The AF3 prediction outputs the scripts consume are not tracked in
this repository.
