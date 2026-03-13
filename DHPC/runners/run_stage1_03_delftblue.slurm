#!/bin/bash
# DelftBlue SLURM template for AFM Stage1 only (case 03).
# Usage: sbatch DHPC/run_stage1_03_delftblue.slurm

#SBATCH --job-name=afm03_stage1
#SBATCH --account=education-me-msc-me
#SBATCH --partition=compute-p2
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/stage1_step2a/slurm_stage1_03_%j.out

set -euo pipefail

module purge
module load julia

REPO_DIR="$HOME/HNODECB"
cd "$REPO_DIR"
mkdir -p logs logs/stage1_step2a

export JULIA_NUM_THREADS=6
export HNODECB_STAGE1_SHARD_COUNT=4
unset HNODECB_STAGE1_SHARD_INDEX

export HNODECB_STAGE1_PREFLIGHT=0
export HNODECB_STAGE1_LOG_EVERY=1
export HNODECB_LR_ADAPT=0
export HNODECB_INF_LOG=1
export HNODECB_LOG_NN_ERR=1
export HNODECB_TRACE_SOLVE=1
export HNODECB_TRACE_SOLVE_LIMIT=50
export HNODECB_STAGE1_AUTOSPAWN=1
export HNODECB_PIPELINE03_AUTOTUNE=0
export HNODECB_JULIA_PROJECT="$REPO_DIR/HPC_env"
export JULIA_PROJECT="$REPO_DIR/HPC_env"

STAGE1_LOG="logs/stage1_step2a/log2_03_step2a_stage1_driver_${SLURM_JOB_ID}.txt"
stdbuf -oL -eL julia --project="$REPO_DIR/HPC_env" runner/stagewise/afm_param_search_stage1_03.jl 2>&1 | tee "$STAGE1_LOG"
