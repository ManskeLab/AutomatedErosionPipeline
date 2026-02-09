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

DATA_DIR=$1
OUT_DIR=$2

INPUT_DIR=/work/manske_lab/jobs/mcp_erosion
export nnUNet_raw=$INPUT_DIR/nnUNet_raw
export nnUNet_preprocessed=$INPUT_DIR/nnUNet_preprocessed
export nnUNet_results=$INPUT_DIR/nnUNet_results

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
#         -device cpu --verbose --save_probabilities

#     nnUNetv2_predict \
#         -d Dataset001_mcp -c 3d_fullres -tr nnUNetTrainerWithAttention \
#         -p nnUNetPlans -f $f \
#         -i "$DATA_DIR" \
#         -o "$OUT_DIR/fold_$f" \
#         -device cpu --verbose --save_probabilities
# done

# echo 

# echo nnUNetv2_ensemble -o $OUT_DIR \
#     -i "$OUT_DIR/fold_0" "$OUT_DIR/fold_1" "$OUT_DIR/fold_2" "$OUT_DIR/fold_3" "$OUT_DIR/fold_4"

# nnUNetv2_ensemble -o $OUT_DIR \
#     -i "$OUT_DIR/fold_0" "$OUT_DIR/fold_1" "$OUT_DIR/fold_2" "$OUT_DIR/fold_3" "$OUT_DIR/fold_4"

# echo

# echo nnUNetv2_apply_postprocessing -i $OUT_DIR \
#     -o $OUT_DIR \
#     -pp_pkl_file /work/manske_lab/jobs/mcp_erosion/nnUNet_results/Dataset001_mcp/nnUNetTrainerWithAttention__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
#     -np 8 -plans_json /work/manske_lab/jobs/mcp_erosion/nnUNet_results/Dataset001_mcp/nnUNetTrainerWithAttention__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json

# nnUNetv2_apply_postprocessing -i $OUT_DIR \
#     -o $OUT_DIR \
#     -pp_pkl_file /work/manske_lab/jobs/mcp_erosion/nnUNet_results/Dataset001_mcp/nnUNetTrainerWithAttention__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
#     -np 8 -plans_json /work/manske_lab/jobs/mcp_erosion/nnUNet_results/Dataset001_mcp/nnUNetTrainerWithAttention__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json