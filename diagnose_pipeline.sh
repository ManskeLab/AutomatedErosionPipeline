#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:15:00
#SBATCH --mem=8GB
#SBATCH --job-name=ERO_DIAG
#SBATCH --output=ERO_DIAG_%j.out

# Report nonzero voxels + spacing/geometry at every pipeline stage for one image.
# Usage:
#   sbatch diagnose_pipeline.sh <WORK_DIR> <INPUT_IMAGE>

source ~/setup_conda.sh
source deactivate
source activate manskelab

SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

WORK_DIR=${1:-/work/manske_lab/jobs/actus_erosion/work}
INPUT_IMAGE=${2:-/work/manske_lab/images/hrpqct/actus/ACTUS_clean/mcp/disease/ACTUS_001/0/ACTUS_001_0_mcp3.nii.gz}

echo "WORK_DIR=$WORK_DIR"
echo "INPUT_IMAGE=$INPUT_IMAGE"
echo

python $SCRIPT_DIR/diagnose_pipeline.py --work-dir "$WORK_DIR" --input "$INPUT_IMAGE"
