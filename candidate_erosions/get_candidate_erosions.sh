#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=EA
#SBATCH --output=EA_%A_%a.out

source ~/setup_conda.sh
source deactivate
source activate manskelab

echo Python version and list of packages:
which python
python -m pip list
echo

SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

# input stripped MCP
INPUT_IMAGE=$1

# registered atlas
ATLAS=$2

# Edge unet output
EDGE=$3

# closed edge unet output
RA_MASK=$4

OUT_DIR=$5

echo python $SCRIPT_DIR/segm_erosion.py --ra $INPUT_IMAGE --atlas $ATLAS --edge $EDGE --ra_mask $RA_MASK --output_dir $OUT_DIR --sr
python $SCRIPT_DIR/segm_erosion.py --ra $INPUT_IMAGE --atlas $ATLAS --edge $EDGE --ra_mask $RA_MASK --output_dir $OUT_DIR --sr

# IF CBCT ONLY
# echo python $SCRIPT_DIR/segm_erosion.py --ra $INPUT_IMAGE --atlas $ATLAS --edge $EDGE --ra_mask $RA_MASK --output_dir $OUT_DIR --sr
# python $SCRIPT_DIR/segm_erosion.py --ra $INPUT_IMAGE --atlas $ATLAS --edge $EDGE --ra_mask $RA_MASK --output_dir $OUT_DIR --sr
