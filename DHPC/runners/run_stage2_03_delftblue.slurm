#!/bin/bash
# DelftBlue SLURM template for AFM Stage2 only (case 03).
# Usage: sbatch DHPC/runners/run_stage2_03_delftblue.slurm

#SBATCH --job-name=afm03_stage2
#SBATCH --account=education-me-msc-me
#SBATCH --partition=compute-p2
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/slurm_stage2_03_%j.out

set -euo pipefail

module purge
module load julia

REPO_DIR="$HOME/HNODECB"
cd "$REPO_DIR"
mkdir -p logs logs/stage2_step2a

export JULIA_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HNODECB_SELFTEST=0
export HNODECB_LR_ADAPT="${HNODECB_LR_ADAPT:-1}"
export HNODECB_LR_MIN="${HNODECB_LR_MIN:-1e-6}"
export HNODECB_LR_MAX="${HNODECB_LR_MAX:-1e-2}"
export HNODECB_LR_ETA="${HNODECB_LR_ETA:-0.05}"
export HNODECB_LR_EMA="${HNODECB_LR_EMA:-0.97}"
export HNODECB_LR_TARGET="${HNODECB_LR_TARGET:-NaN}"
export HNODECB_STAGE2_LR_FORCE="${HNODECB_STAGE2_LR_FORCE:-NaN}"
export HNODECB_STAGE2_OPTIMIZER="${HNODECB_STAGE2_OPTIMIZER:-amsgrad}"
export HNODECB_STAGE2_GRAD_SCALE_P_NET="${HNODECB_STAGE2_GRAD_SCALE_P_NET:-0.2}"
export HNODECB_STAGE2_GRAD_SCALE_MECH="${HNODECB_STAGE2_GRAD_SCALE_MECH:-1.0}"
export HNODECB_STAGE2_GROUP_ADAPT="${HNODECB_STAGE2_GROUP_ADAPT:-1}"
export HNODECB_STAGE2_GROUP_ETA="${HNODECB_STAGE2_GROUP_ETA:-0.15}"
export HNODECB_STAGE2_VERIFY_ADAM_DIR="${HNODECB_STAGE2_VERIFY_ADAM_DIR:-1}"
export HNODECB_STAGE2_LOG_EVERY=1
export HNODECB_STAGE2_AUTOSPAWN="${HNODECB_STAGE2_AUTOSPAWN:-1}"
export HNODECB_STAGE2_SHARD_COUNT=2
export HNODECB_STAGE2_INPUT_TOPK="${HNODECB_STAGE2_INPUT_TOPK:-9}"
export HNODECB_JULIA_PROJECT="$REPO_DIR/HPC_env"
export JULIA_PROJECT="$REPO_DIR/HPC_env"

STAGE2_DRIVER_LOG="logs/stage2_step2a/log2_03_step2a_stage2_driver_${SLURM_JOB_ID}.txt"
set +e
stdbuf -oL -eL julia --project="$REPO_DIR/HPC_env" runner/stagewise/afm_param_search_stage2_03.jl 2>&1 | tee "$STAGE2_DRIVER_LOG"
STAGE2_RC=${PIPESTATUS[0]}
set -e

STAGE2_MEM_LOG="logs/stage2_step2a/log2_03_step2a_stage2_mem_${SLURM_JOB_ID}.txt"
{
  echo "=== Stage2 Resource Summary ==="
  echo "job_id=${SLURM_JOB_ID}"
  date
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName%30,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,MaxVMSize -P
  else
    echo "sacct not found; cannot collect MaxRSS."
  fi
} | tee "$STAGE2_MEM_LOG"

exit "$STAGE2_RC"
