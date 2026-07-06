#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=ERO_PIPE
#SBATCH --output=ERO_PIPE_%j.out

# Orchestrator job: walks the input dataset and submits the full per-image
# pipeline (strip -> edge/closed -> reg -> candidate -> predict -> combine)
# with SLURM dependencies. This job itself is light (it only calls sbatch).
#
# Usage:
#   sbatch run_pipeline.sh \
#       --input-dir /work/manske_lab/images/hrpqct/actus/ACTUS_clean/mcp/disease \
#       --work-dir  /work/manske_lab/jobs/actus_erosion/work \
#       --out-dir   /work/manske_lab/images/hrpqct/actus/ACTUS_clean/mcp/disease_erosions
#
# Everything after run_pipeline.sh is forwarded to run_pipeline.py, so all its
# flags work (--subjects, --mcps, --timepoints, --sr, --dry-run, ...).

source ~/setup_conda.sh
source deactivate
source activate manskelab

SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

echo python $SCRIPT_DIR/run_pipeline.py "$@"
python $SCRIPT_DIR/run_pipeline.py "$@"
