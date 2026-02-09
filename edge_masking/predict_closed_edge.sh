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

INPUT_IMAGE=$1
INPUT_EDGE=$2
OUT_DIR=$3

INPUT_DIR=/work/manske_lab/jobs/mcp_closed_edge
export nnUNet_raw=$INPUT_DIR/nnUNet_raw
export nnUNet_preprocessed=$INPUT_DIR/nnUNet_preprocessed
export nnUNet_results=$INPUT_DIR/nnUNet_results

INPUT_NAME=${INPUT_IMAGE##*/}
INPUT_NAME=${INPUT_NAME%%.*}
DATA_DIR=$INPUT_DIR/Dataset001_mcp/imagesTs/$INPUT_NAME
mkdir -p $DATA_DIR
cp $INPUT_IMAGE $DATA_DIR/${INPUT_NAME}_0000.nii.gz
cp $INPUT_EDGE $DATA_DIR/${INPUT_NAME}_0001.nii.gz

mkdir $OUT_DIR

echo nnUNetv2_predict \
    -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainerWithAttention \
    -p nnUNetPlans -f all \
    -i "$DATA_DIR" \
    -o "$OUT_DIR" \
    -device cpu --verbose

nnUNetv2_predict \
    -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainerWithAttention \
    -p nnUNetPlans -f all \
    -i "$DATA_DIR" \
    -o "$OUT_DIR" \
    -device cpu --verbose

# for f in 0 1 2 3 4; do
#     mkdir $OUT_DIR/fold_$f

#     echo nnUNetv2_predict \
#         -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainerWithAttention \
#         -p nnUNetPlans -f $f \
#         -i "$DATA_DIR" \
#         -o "$OUT_DIR/fold_$f" \
#         -device cpu --verbose

#     nnUNetv2_predict \
#         -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainerWithAttention \
#         -p nnUNetPlans -f $f \
#         -i "$DATA_DIR" \
#         -o "$OUT_DIR/fold_$f" \
#         -device cpu --verbose
# done

# echo nnUNetv2_ensemble -o $OUT_DIR \
#     -f "$OUT_DIR/fold_0" "$OUT_DIR/fold_1" "$OUT_DIR/fold_2" "$OUT_DIR/fold_3" "$OUT_DIR/fold_4"
# nnUNetv2_ensemble -o $OUT_DIR \
#     -f "$OUT_DIR/fold_0" "$OUT_DIR/fold_1" "$OUT_DIR/fold_2" "$OUT_DIR/fold_3" "$OUT_DIR/fold_4"

# echo nnUNetv2_apply_postprocessing -i $OUT_DIR -o $OUT_DIR -pp_pkl_file /work/manske_lab/jobs/mcp_closed_edge/nnUNet_results/Dataset001_mcp/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /work/manske_lab/jobs/mcp_closed_edge/nnUNet_results/Dataset001_mcp/nnUNetTrainerWithAttention__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json
# nnUNetv2_apply_postprocessing -i $OUT_DIR -o $OUT_DIR -pp_pkl_file /work/manske_lab/jobs/mcp_closed_edge/nnUNet_results/Dataset001_mcp/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /work/manske_lab/jobs/mcp_closed_edge/nnUNet_results/Dataset001_mcp/nnUNetTrainerWithAttention__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json

# echo nnUNetv2_predict -d Dataset001_hand -i $DATA_DIR -o $OUT_DIR -f 4 -tr nnUNetTrainer -c 3d_cascade_fullres_large -p nnUNetPlans -prev_stage_predictions $OUT_DIR --verbose -device cpu
# nnUNetv2_predict -d Dataset001_hand -i $DATA_DIR -o $OUT_DIR -f 4 -tr nnUNetTrainer -c 3d_cascade_fullres_large -p nnUNetPlans -prev_stage_predictions $OUT_DIR --verbose -device cpu
# echo nnUNetv2_apply_postprocessing -i $OUT_DIR -o $OUT_DIR -pp_pkl_file /work/manske_lab/jobs/cbct_nnunet_hand/nnUNet_results/Dataset001_hand/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres_large/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /work/manske_lab/jobs/cbct_nnunet_hand/nnUNet_results/Dataset001_hand/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres_large/crossval_results_folds_0_1_2_3_4/plans.json
# nnUNetv2_apply_postprocessing -i $OUT_DIR -o $OUT_DIR -pp_pkl_file /work/manske_lab/jobs/cbct_nnunet_hand/nnUNet_results/Dataset001_hand/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres_large/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /work/manske_lab/jobs/cbct_nnunet_hand/nnUNet_results/Dataset001_hand/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres_large/crossval_results_folds_0_1_2_3_4/plans.json