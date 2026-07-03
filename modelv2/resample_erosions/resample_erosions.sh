#!/bin/bash
#SBATCH --job-name=resample_erosions
#SBATCH --partition=cpu2019
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=resample_erosions_%j.out
#SBATCH --error=resample_erosions_%j.err

set -euo pipefail

# --- paths (edit if needed) -------------------------------------------------
IMAGES_TR=/work/manske_lab/jobs/mcp_erosion/nnUNet_raw/Dataset001_mcp/labelsTr
MCP_ROOT=/work/manske_lab/images/hrpqct/rair/rair_mcp
OUT_DIR=/work/manske_lab/images/hrpqct/rair/erosions_full
MATCH_CSV=/work/manske_lab/images/hrpqct/rair/nnUNet_matches_clean.csv

SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

# --- environment ------------------------------------------------------------
# Uses the conda `base` env (SimpleITK must be installed there).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

# --- run --------------------------------------------------------------------
python "${SCRIPT_DIR}/resample_erosions.py" \
    --imagesTr   "${IMAGES_TR}" \
    --mcp-root   "${MCP_ROOT}" \
    --out-dir    "${OUT_DIR}" \
    --csv        "${MATCH_CSV}" \
    --interp     nearest

echo "Finished on $(date)"
