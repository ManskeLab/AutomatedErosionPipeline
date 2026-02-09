import argparse
import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask a NIfTI image using a binary mask"
    )
    parser.add_argument("--image", required=True, help="Input image (.nii or .nii.gz)")
    parser.add_argument("--mask", required=True, help="Input mask (.nii or .nii.gz)")
    parser.add_argument("--out", required=True, help="Output masked image path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image and mask
    img = nib.load(args.image)
    mask = nib.load(args.mask)

    img_data = img.get_fdata()
    mask_data = mask.get_fdata()

    # Basic sanity check
    if img_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: image {img_data.shape} vs mask {mask_data.shape}"
        )

    # Apply mask (keep where mask > 0)
    masked = img_data * (mask_data > 0)

    # Save output (preserve affine + header)
    out_img = nib.Nifti1Image(masked, img.affine, img.header)
    nib.save(out_img, args.out)


if __name__ == "__main__":
    main()