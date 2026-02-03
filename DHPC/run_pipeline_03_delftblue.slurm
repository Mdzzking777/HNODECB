#!/bin/bash
# DelftBlue SLURM template for AFM pipeline (Stage1 shards + Stage2)
# Usage: sbatch run_pipeline_03_delftblue.slurm

#SBATCH --job-name=afm03_pipeline
#SBATCH --account=education-me-msc-me
#SBATCH --partition=compute-p2
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=logs/slurm_pipeline03_%j.out

set -euo pipefail

module purge
module load julia

# Project root (update if different)
REPO_DIR="$HOME/HNODECB"
cd "$REPO_DIR"

# Threads for Julia (within one process)
export JULIA_NUM_THREADS=6

# Stage1 sharding (multi-process on one node)
export HNODECB_STAGE1_SHARD_COUNT=4
unset HNODECB_STAGE1_SHARD_INDEX

# Optional runtime toggles
export HNODECB_STAGE1_PREFLIGHT=0
export HNODECB_STAGE1_LOG_EVERY=1
export HNODECB_INF_LOG=1
export HNODECB_LOG_NN_ERR=1
export HNODECB_TRACE_SOLVE=1
export HNODECB_TRACE_SOLVE_LIMIT=50
export HNODECB_PIPELINE03_AUTOSPAWN=1
export HNODECB_PIPELINE03_AUTOTUNE=0
export HNODECB_JULIA_PROJECT="$REPO_DIR/HPC_env"
export JULIA_PROJECT="$REPO_DIR/HPC_env"

# Run pipeline (Stage1 -> Stage2)
julia --project="$REPO_DIR/HPC_env" runner/afm_param_search_pipeline_03.jl
