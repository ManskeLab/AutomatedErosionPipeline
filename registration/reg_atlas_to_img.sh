#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --mem=20GB
#SBATCH --job-name=RA_DR
#SBATCH --output=RA_DR_%A_%a.out

module load StdEnv/2020
module load gcc/9.3.0
module load ants

SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=($SCRIPT_DIR)
SCRIPT_DIR=$(dirname ${SCRIPT_DIR[0]})

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=48

IN_IMG=$1

# PP or MC mask
IN_MASK=$2

ATLAS=$3
OUTPUT_DIR=$4

# PP or MC
BONE=$5

echo Input Image: $IN_IMG
echo Atlas: $ATLAS

SCRATCH=/scratch/$SLURM_JOBID

IN_IMG_TEMP=$SCRATCH/masked_img.nii.gz

ImageMath 3 $IN_IMG_TEMP m $IN_IMG $IN_MASK

ATLAS_TEMP=$SCRATCH/atlas_initial_align.nii.gz
cp $ATLAS $ATLAS_TEMP

# Only for CBCT (flips left handed, hrpqct already flipped)
# filename=${IN_IMG##*/}
# cbct_filename=$(find /work/manske_lab/images/cbct/extrem/hand/actus/actus_mcps/ -name "*${filename}*" | head -n 1)

# if [[ "$filename" == *"_R"* ]]; then
#     python $SCRIPT_DIR/initial_atlas_alignment.py $ATLAS $ATLAS_TEMP
# else
#     python $SCRIPT_DIR/initial_atlas_alignment.py $ATLAS $ATLAS_TEMP -f
# fi

filename=${IN_IMG##*/}

filename=${filename%%.*}
echo File basename: $filename
echo

ImageMath 3 $IN_IMG_TEMP HistogramMatch $IN_IMG_TEMP $ATLAS_TEMP 255 64 1

antsRegistration \
	--verbose 1 \
    --dimensionality 3 \
    --float 1 \
    --collapse-output-transforms 1 \
    --output [$SCRATCH/field_mesh_Atlas_RA_$filename, $SCRATCH/${filename}_ATLAS_WARP.nii.gz]\
    --interpolation BSpline[ 5 ] \
    --use-histogram-matching 1 \
    --winsorize-image-intensities [ 0.8, 1 ] \
    --initial-moving-transform [ $IN_IMG_TEMP, $ATLAS_TEMP, 1 ] \
    --transform Similarity[0.1] \
    --metric MI[ $IN_IMG_TEMP, $ATLAS_TEMP, 1, 32, Regular, 0.25 ]\
    --convergence [ 150x100x50x0, 1e-6, 10 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 4x2x1x0vox \
    --transform Affine[ 0.1 ] \
    --metric MI[ $IN_IMG_TEMP, $ATLAS_TEMP, 1, 32, Regular, 0.25 ] \
    --convergence [ 150x100x50x0, 1e-6, 10 ] \
    --shrink-factors 6x4x2x1 \
    --smoothing-sigmas 4x2x1x0vox \
    --transform SyN[0.2, 3, 0.25] \
    --metric MI[ $IN_IMG_TEMP, $ATLAS_TEMP, 1, 32, Regular, 0.25 ]\
    --convergence [ 120x100x70, 1e-6, 10 ] \
    --shrink-factors 16x8x4 \
    --smoothing-sigmas 2x1x0vox 
echo

echo Transforming image to current template...

antsApplyTransforms \
    --verbose 1 \
	-d 3 \
	--float 1 \
	-i $ATLAS_TEMP \
	-o $OUTPUT_DIR/ATLAS_TO_${filename}_${BONE}.nii.gz \
	-r $IN_IMG \
	-t $SCRATCH/field_mesh_Atlas_RA_$filename*1Warp.nii.gz \
	-t $SCRATCH/field_mesh_Atlas_RA_$filename*.mat