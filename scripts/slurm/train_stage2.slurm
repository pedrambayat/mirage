#!/bin/bash
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=stage2
#SBATCH --output=slurm-stage2-%j.out
set -euo pipefail
ESM_PY="$HOME/miniconda3/envs/esm/bin/python"
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage
"$ESM_PY" scripts/train_stage2.py \
  --pairs data/staged/sabdab/sabdab_pairs.csv \
  --perres data/staged/sabdab/perres.npz --perres-keys data/staged/sabdab/perres_keys.txt \
  --pooled data/staged/sabdab/embeddings.npy --pooled-keys data/staged/sabdab/keys.txt \
  --out data/staged/sabdab/stage2_oof_scores.csv --seed 20260601
