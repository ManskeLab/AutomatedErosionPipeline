#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=EA
#SBATCH --output=EA_%A_%a.out

# Per-bone candidate erosion generation. Same as get_candidate_erosions.sh but
# the SR flag is an explicit argument (HR-pQCT -> "nosr", SR-CBCT/CBCT -> "sr").

source ~/setup_conda.sh
source deactivate
source activate manskelab

SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

INPUT_IMAGE=$1   # stripped MCP
ATLAS=$2         # registered atlas (MC or PP)
EDGE=$3          # edge unet output
RA_MASK=$4       # closed edge unet output
OUT_DIR=$5       # per-bone output dir
SR=${6:-nosr}    # "sr" or "nosr"

mkdir -p $OUT_DIR

SR_ARG=""
if [ "$SR" == "sr" ]; then
    SR_ARG="--sr"
fi

echo python $SCRIPT_DIR/segm_erosion.py --ra $INPUT_IMAGE --atlas $ATLAS --edge $EDGE --ra_mask $RA_MASK --output_dir $OUT_DIR $SR_ARG
python $SCRIPT_DIR/segm_erosion.py --ra $INPUT_IMAGE --atlas $ATLAS --edge $EDGE --ra_mask $RA_MASK --output_dir $OUT_DIR $SR_ARG
