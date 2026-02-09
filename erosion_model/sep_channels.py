import SimpleITK as sitk
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('INPUT_IMAGE', type=str)
parser.add_argument('INPUT_NAME', type=str)
parser.add_argument('INPUT_IND_DIR', type=str)

args = parser.parse_args()

img_4d = sitk.ReadImage(args.INPUT_IMAGE)  # Assumes shape [X, Y, Z, C]
array = sitk.GetArrayFromImage(img_4d)     # Shape: [Z, Y, X, C]

if array.shape[-1] != 3:
    raise ValueError("Expected 3-channel image (last dimension should be 3).")

for c in range(3):
    channel_arr = array[..., c]  # shape: [Z, Y, X]
    img_3d = sitk.GetImageFromArray(channel_arr)
    img_3d.SetSpacing(img_4d.GetSpacing()[:3])
    img_3d.SetOrigin(img_4d.GetOrigin()[:3])
    img_3d.SetDirection(img_4d.GetDirection()[:9])  # 3x3

    out_path = os.path.join(args.INPUT_IND_DIR, f"{args.INPUT_NAME}_000{c}.nii.gz")
    sitk.WriteImage(img_3d, out_path)
    print(f"Saved: {out_path}")

# Generate metadata JSON for nnUNetv2
sample_img = sitk.ReadImage(os.path.join(args.INPUT_IND_DIR, f"{args.INPUT_NAME}_0000.nii.gz"))
spacing = list(sample_img.GetSpacing())  # [x, y, z]
size = list(sample_img.GetSize())        # [x, y, z]

metadata = {
    "modality": {
        "0": "CT",
        "1": "Edge",
        "2": "Atlas"
    },
    "spacing": spacing[::-1],  # nnUNet expects [z, y, x]
    "shape": size[::-1]
}

json_path = os.path.join(args.INPUT_IND_DIR, f"{args.INPUT_NAME}.json")
with open(json_path, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata JSON saved to: {json_path}")