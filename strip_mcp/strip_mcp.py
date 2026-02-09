import argparse
import os
import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask a NIfTI image using a label mask"
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

    if img_data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: image {img_data.shape} vs mask {mask_data.shape}"
        )

    # Output directory + base filename
    out_dir = os.path.dirname(args.out)
    out_name = os.path.basename(args.out)

    if out_dir == "":
        out_dir = "."

    os.makedirs(out_dir, exist_ok=True)

    # ---- full masked image (mask > 0) ----
    masked_all = img_data * (mask_data > 0)
    nib.save(
        nib.Nifti1Image(masked_all, img.affine, img.header),
        args.out,
    )

    # ---- MC (label == 1) ----
    mc_masked = img_data * (mask_data == 1)
    mc_out = os.path.join(out_dir, f"MC_{out_name}")
    nib.save(
        nib.Nifti1Image(mc_masked, img.affine, img.header),
        mc_out,
    )

    # ---- PP (label == 2) ----
    pp_masked = img_data * (mask_data == 2)
    pp_out = os.path.join(out_dir, f"PP_{out_name}")
    nib.save(
        nib.Nifti1Image(pp_masked, img.affine, img.header),
        pp_out,
    )


if __name__ == "__main__":
    main()