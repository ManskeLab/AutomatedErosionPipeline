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
DATA_DIR=$INPUT_DIR/nnUNet_raw/Dataset001_hand/imagesTs/$INPUT_NAME
mkdir -p $DATA_DIR $OUT_DIR

# The whole pipeline runs at 1.0mm spacing (metadata relabel, NO resampling)
# because every nnUNet model was trained on 1.0mm data. Native HR-pQCT spacing
# makes nnUNet rescale the image ~16x and produce garbage masks. We make a
# clean-named 1mm copy and strip THAT, so the stripped image (and every later
# stage) is 1mm too. The true affine is stamped back on at the combine step.
IN1MM_DIR=$OUT_DIR/_in1mm
mkdir -p $IN1MM_DIR
IMG_1MM=$IN1MM_DIR/${INPUT_NAME}.nii.gz
cp $INPUT_IMAGE $IMG_1MM
python $SCRIPT_DIR/config_metadata.py $IMG_1MM --spacing 1.0 1.0 1.0

cp $IMG_1MM $DATA_DIR/${INPUT_NAME}_0000.nii.gz

echo nnUNetv2_predict \
    -d Dataset001_hand -c 3d_fullres -tr nnUNetTrainer \
    -p nnUNetPlans -f all \
    -i "$DATA_DIR" \
    -o "$OUT_DIR" \
    -device cpu --verbose

nnUNetv2_predict \
    -d Dataset001_hand -c 3d_fullres -tr nnUNetTrainer \
    -p nnUNetPlans -f all \
    -i "$DATA_DIR" \
    -o "$OUT_DIR" \
    -device cpu --verbose

OUT_MASK=$OUT_DIR/$INPUT_NAME.nii.gz
# Strip the 1mm image (not the raw one) so stripped_<name>.nii.gz is 1mm.
python $SCRIPT_DIR/strip_mcp.py --image $IMG_1MM --mask $OUT_MASK --out $OUT_DIR
mv $OUT_MASK $OUT_DIR/mask_${INPUT_NAME}.nii.gz