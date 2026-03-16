#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --time=2:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=HS_P
#SBATCH --output=HS_P_%j.out

# set_spacing_1mm.sh
# Set all .nii.gz files in a directory to 1mm isotropic spacing (metadata only, no resampling).
# Usage: bash set_spacing_1mm.sh <directory>

set -euo pipefail

DIR="${1:-}"

if [[ -z "$DIR" ]]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

if [[ ! -d "$DIR" ]]; then
    echo "Error: '$DIR' is not a directory."
    exit 1
fi

FILES=("$DIR"/*.nii.gz)

if [[ ! -e "${FILES[0]}" ]]; then
    echo "No .nii.gz files found in '$DIR'."
    exit 1
fi

echo "Setting spacing to 1.0 x 1.0 x 1.0 mm for all .nii.gz in: $DIR"
echo "------------------------------------------------------------"

for f in "${FILES[@]}"; do
    echo "Processing: $(basename "$f")"
    python3 - "$f" <<'PYEOF'
import sys
import SimpleITK as sitk

path = sys.argv[1]
img = sitk.ReadImage(path)
ndim = img.GetDimension()

old = img.GetSpacing()
new = tuple([1.0] * ndim)

img.SetSpacing(new)
sitk.WriteImage(img, path)

print(f"  {old} -> {new}")
PYEOF
done

echo "------------------------------------------------------------"
echo "Done."