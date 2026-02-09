# Atlas-Guided Erosion Segmentation

## Prerequisites

- Install our in-house customized **nnUNet**
- Soft-tissue–stripped MCP joint images from **HR-pQCT** or **SR-CBCT**
  - With **MC** and **PP** bone masks
- Atlas, located in:
  - `<PATH_TO_ATLAS>`

### Model Checkpoints

#### MCP Stripping
  /work/manske_lab/jobs/mcp_nnunet/nnUNet_results/Dataset001_hand/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth
#### Bone edge mask
  /work/manske_lab/jobs/mcp_edge/nnUNet_results/Dataset001_mcp/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth
#### Bone closed edge mask  
  /work/manske_lab/jobs/mcp_closed_edge/nnUNet_results/Dataset001_mcp/nnUNetTrainerWithAttention__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth
#### Erosion segmentation
  /work/manske_lab/jobs/mcp_erosion/nnUNet_results/Dataset001_mcp/nnUNetTrainerWithAttention__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth

---

## Pipeline Steps

### 1. Strip MCP Image
```bash
sbatch strip_mcp/predict_mcp.sh \
  $MCP_IMAGE \
  $OUTPUT_DIR
```
Use the stripped image (output of above script $OUTPUT_DIR/stripped_<input_image>.nii.gz) as MCP_IMAGE.
### 2. Run both edge segmentation scripts to get closed edge and bone edge masks
```bash
sbatch edge_masking/predict_edge.sh \
  $MCP_IMAGE \
  $OUTPUT_DIR
```
### 3. Get closed edge segmentation
```bash
sbatch edge_masking/predict_closed_edge.sh \
  $MCP_IMAGE \
  $EDGE_MASK \
  $OUTPUT_DIR
```
### 4. Run registrations
```bash
sbatch registration/reg_atlas_to_img.sh \
  $MCP_IMAGE \
  $MC_OR_PP_BONE_MASK \
  $ATLAS \
  $OUTPUT_DIR \
  $BONE_NAME
```
### 5. Get candidate erosions
```bash
sbatch candidate_erosions/get_candidate_erosions.sh \
  $MCP_IMAGE \
  $REGISTERED_ATLAS \
  $EDGE_MASK \
  $CLOSED_EDGE_MASK \
  $OUTPUT_DIR
```
### 6.. Go through erosions candidates and create a folder of erosion ROIs.
### 7. Run the folder through erosion nnUNet.
```bash
sbatch erosion_model/batch_predict.sh \
  $PATH_TO_EROSIONS_ROI_DIR
```
  
