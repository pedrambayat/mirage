#!/bin/bash
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=esm_embed
#SBATCH --output=slurm-%j.out
set -euo pipefail
# Run in the dedicated ESM env (torch + fair-esm). Create once:
#   conda create -n esm python=3.11 -y
#   ~/miniconda3/envs/esm/bin/pip install torch fair-esm numpy
# Call the env's python by absolute path (robust in a non-interactive batch
# shell where `conda activate` may not be available on PATH).
ESM_PY="$HOME/miniconda3/envs/esm/bin/python"
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
"$ESM_PY" scripts/embed_sequences.py \
  --manifest data/staged/sabdab/sabdab_unique_seqs.txt \
  --out-embeddings data/staged/sabdab/embeddings.npy \
  --out-keys data/staged/sabdab/keys.txt
