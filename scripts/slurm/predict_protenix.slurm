#!/bin/bash
#SBATCH --job-name=mirage_protenix
# Account, partition, qos, gres, time, array, and log paths are all supplied
# by `ProtenixPosePredictor.submit()` as sbatch CLI flags. Defaults live
# there, not here, so changing partition/time does not require editing this
# script.
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <manifest.tsv> <chunk_size> <runner_path>" >&2
    # sbatch copies this script to its spool dir, so BASH_SOURCE-based path
    # resolution does not find the sibling runner. The submitter passes the
    # runner's absolute path explicitly instead.
    exit 2
fi
MANIFEST="$1"
CHUNK_SIZE="$2"
RUNNER="$3"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "[predict_protenix] host=$(hostname) task=${TASK_ID} chunk_size=${CHUNK_SIZE}"
echo "[predict_protenix] manifest=${MANIFEST}"
echo "[predict_protenix] runner=${RUNNER}"

# Blackwell env vars required before conda activate (protenix reads them at
# import time). Do NOT remove — cuequivariance kernels hang without these.
export PROTENIX_ROOT_DIR=/vast/projects/dbgoodma/goodman-laboratory/pbayat/.protenix
export LAYERNORM_TYPE=native
export MMSEQS_SERVICE_HOST_URL=https://api.colabfold.com

source /vast/home/p/pbayat/miniconda3/etc/profile.d/conda.sh
conda activate protenix

nvidia-smi -L || true

exec python3 "${RUNNER}" "${MANIFEST}" "${TASK_ID}" "${CHUNK_SIZE}"
