#!/bin/bash
# Precompute ColabFold MSAs once per unique sequence, sharded across an array.
#
# Usage (array index = shard index):
#   sbatch --array=0-7%8 scripts/slurm/precompute_msa.slurm \
#          <unique_manifest> <cache_dir> <work_dir> <n_shards>
#
# NOTE: this is network-bound (MMseqs2 calls to api.colabfold.com), NOT GPU work.
# Run it on a partition with OUTBOUND network egress to api.colabfold.com. The
# PARCC login node and dgx-b200 nodes have egress; genoa CPU nodes are UNVERIFIED.
# For a one-off campaign, running the python driver directly on the login node in
# background shards is usually simpler and wastes no GPU.
#SBATCH --account=dbgoodma-goodman-laboratory
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --time=06:00:00
#SBATCH --job-name=mirage_msa
set -uo pipefail

MANIFEST="$1"; CACHE_DIR="$2"; WORK_DIR="$3"; N_SHARDS="$4"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

export PROTENIX_ROOT_DIR=/vast/projects/dbgoodma/goodman-laboratory/pbayat/.protenix
export LAYERNORM_TYPE=native
export MMSEQS_SERVICE_HOST_URL=https://api.colabfold.com
source /vast/home/p/pbayat/miniconda3/etc/profile.d/conda.sh
conda activate protenix
cd /vast/projects/dbgoodma/goodman-laboratory/pbayat/binder-discrimination/mirage

echo "[precompute_msa.slurm] host=$(hostname) shard=${SHARD}/${N_SHARDS}"
exec python3 scripts/precompute_protenix_msa.py \
  --unique-manifest "$MANIFEST" --cache-dir "$CACHE_DIR" \
  --work-dir "$WORK_DIR" --shard-index "$SHARD" --n-shards "$N_SHARDS"
