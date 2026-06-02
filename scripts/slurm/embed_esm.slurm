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
#   conda create -n esm python=3.11 -y && conda activate esm
#   pip install torch fair-esm numpy
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esm
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
python scripts/embed_sequences.py \
  --manifest data/staged/sabdab/sabdab_unique_seqs.txt \
  --out-embeddings data/staged/sabdab/embeddings.npy \
  --out-keys data/staged/sabdab/keys.txt
