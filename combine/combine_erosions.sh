#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --job-name=ERO_COMB
#SBATCH --output=ERO_COMB_%j.out

source ~/setup_conda.sh
source deactivate
source activate manskelab

SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

PRED_DIR=$1    # per-image erosion predictions
REF_IMAGE=$2   # original input MCP image (defines output grid)
OUT_FILE=$3    # output combined mask

echo python $SCRIPT_DIR/combine_erosions.py --pred-dir $PRED_DIR --ref $REF_IMAGE --out $OUT_FILE
python $SCRIPT_DIR/combine_erosions.py --pred-dir $PRED_DIR --ref $REF_IMAGE --out $OUT_FILE
