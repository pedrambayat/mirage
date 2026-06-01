#!/bin/bash
#SBATCH --job-name=abdisc_af2m
# Account, partition, gres, time, array, and log paths are all supplied by
# `AF2MPosePredictor.submit()` as sbatch CLI flags. Defaults live there, not
# here, so changing the partition does not require editing this script.
set -euo pipefail

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

echo "[predict_af2m] host=$(hostname) task=${TASK_ID} chunk_size=${CHUNK_SIZE}"
echo "[predict_af2m] manifest=${MANIFEST}"
echo "[predict_af2m] runner=${RUNNER}"
nvidia-smi -L || true

exec python3 "${RUNNER}" "${MANIFEST}" "${TASK_ID}" "${CHUNK_SIZE}"
