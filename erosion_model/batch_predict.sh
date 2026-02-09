#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=1GB
#SBATCH --job-name=HS
#SBATCH --output=HS_%j.out

echo Running batch_predict.sh...
echo

source ~/setup_conda.sh
source deactivate
source activate manskelab

# Set variables
SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

INPUT_DIR=/work/manske_lab/jobs/mcp_erosion/nnUNet_raw/Dataset001_mcp/imagesTs
mkdir $INPUT_DIR

SOURCE_DIR=/work/manske_lab/images/hrpqct/actus/actus_erosion_candidates
ID=HR

for INPUT_IMAGE in $SOURCE_DIR/*.nii*; do

    INPUT_NAME=${INPUT_IMAGE##*/}
    INPUT_NAME=${INPUT_NAME%%.*}

    INPUT_IND_DIR=$INPUT_DIR/${INPUT_NAME}_${ID}

    echo $INPUT_IND_DIR
    mkdir $INPUT_IND_DIR
    echo ${INPUT_IND_DIR}_pred
    mkdir ${INPUT_IND_DIR}_pred

    # rsync $INPUT_IMAGE $INPUT_IND_DIR/${INPUT_NAME}_0000.nii.gz

    echo python $SCRIPT_DIR/sep_channels.py $INPUT_IMAGE $INPUT_NAME $INPUT_IND_DIR
    python $SCRIPT_DIR/sep_channels.py $INPUT_IMAGE $INPUT_NAME $INPUT_IND_DIR

    echo sbatch $SCRIPT_DIR/predict.sh $INPUT_IND_DIR ${INPUT_IND_DIR}_pred
    sbatch $SCRIPT_DIR/predict.sh $INPUT_IND_DIR ${INPUT_IND_DIR}_pred

done

echo $SOURCE_DIR