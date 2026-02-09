#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=HS_P
#SBATCH --output=HS_P_%j.out

echo Running predict.sh...
echo

echo start initialization

source ~/setup_conda.sh
source deactivate
source activate attention_nnunet

echo Python version and list of packages:
which python
python -m pip list
echo

# Set variables
SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

INPUT_IMAGE=$1
OUT_DIR=$2

INPUT_DIR=/work/manske_lab/jobs/mcp_nnunet
export nnUNet_raw=$INPUT_DIR/nnUNet_raw
export nnUNet_preprocessed=$INPUT_DIR/nnUNet_preprocessed
export nnUNet_results=$INPUT_DIR/nnUNet_results

INPUT_NAME=${INPUT_IMAGE##*/}
INPUT_NAME=${INPUT_NAME%%.*}
DATA_DIR=$INPUT_DIR/Dataset001_mcp/imagesTs/$INPUT_NAME
mkdir -p $DATA_DIR
cp $INPUT_IMAGE $DATA_DIR/${INPUT_NAME}_0000.nii.gz

mkdir -p $OUT_DIR

echo nnUNetv2_predict \
    -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainer \
    -p nnUNetPlans -f all \
    -i "$DATA_DIR" \
    -o "$OUT_DIR" \
    -device cpu --verbose

nnUNetv2_predict \
    -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainer \
    -p nnUNetPlans -f all \
    -i "$DATA_DIR" \
    -o "$OUT_DIR" \
    -device cpu --verbose

OUT_MASK=$(ls "$OUT_DIR"/*.nii.gz 2>/dev/null | head -n 1)
python $SCRIPT_DIR/strip_mcp.py $INPUT_NAME $OUT_MASK $OUT_DIR/stripped_${INPUT_NAME}.nii.gz