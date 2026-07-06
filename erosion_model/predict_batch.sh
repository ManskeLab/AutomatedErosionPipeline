#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=ERO_P
#SBATCH --output=ERO_P_%j.out

# Batched erosion prediction for one MCP image: takes the MC and PP candidate
# erosion dirs, splits each 3-channel ROI into nnUNet channels under one imagesTs
# folder (case names encode bone + erosion index), then runs nnUNet ONCE.

echo start initialization
source ~/setup_conda.sh
source deactivate
source activate attention_nnunet

SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

CAND_MC_DIR=$1     # candidate erosions for MC
CAND_PP_DIR=$2     # candidate erosions for PP
ERO_INPUT_DIR=$3   # scratch: split-channel nnUNet input folder (one image)
OUT_PRED_DIR=$4    # nnUNet predictions out
KEY=$5             # image key, e.g. ACTUS_001_0_mcp1

INPUT_DIR=/work/manske_lab/jobs/mcp_erosion
export nnUNet_raw=$INPUT_DIR/nnUNet_raw
export nnUNet_preprocessed=$INPUT_DIR/nnUNet_preprocessed
export nnUNet_results=$INPUT_DIR/nnUNet_results

rm -rf $ERO_INPUT_DIR $OUT_PRED_DIR
mkdir -p $ERO_INPUT_DIR $OUT_PRED_DIR

# Split every ROI (both bones) into channels with a unique, decodable case name.
for BONE in MC PP; do
    if [ "$BONE" == "MC" ]; then DIR=$CAND_MC_DIR; else DIR=$CAND_PP_DIR; fi
    for INPUT_IMAGE in $DIR/*_input*.nii.gz; do
        [ -e "$INPUT_IMAGE" ] || continue
        BASE=$(basename $INPUT_IMAGE)
        ERO=$(echo $BASE | grep -oE 'erosion[0-9]+' | head -n1)
        CASE=${KEY}_${BONE}_${ERO}
        echo "  sep_channels: $BASE -> $CASE"
        python $SCRIPT_DIR/sep_channels.py $INPUT_IMAGE $CASE $ERO_INPUT_DIR
    done
done

# Nothing to predict (image had no erosions) -> leave OUT_PRED_DIR empty.
if ! ls $ERO_INPUT_DIR/*_0000.nii.gz >/dev/null 2>&1; then
    echo "No ROIs for $KEY; nothing to predict."
    exit 0
fi

echo nnUNetv2_predict -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainerWithAttention \
    -p nnUNetPlans -f all -i "$ERO_INPUT_DIR" -o "$OUT_PRED_DIR" -device cpu --verbose

nnUNetv2_predict \
    -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainerWithAttention \
    -p nnUNetPlans -f all \
    -i "$ERO_INPUT_DIR" \
    -o "$OUT_PRED_DIR" \
    -device cpu --verbose
