import SimpleITK as sitk
import numpy as np
import argparse
import os

def get_image_center(image):
    """
    Compute the physical center of an image in world coordinates.
    """
    size = np.array(image.GetSize())  # Image size in voxels
    spacing = np.array(image.GetSpacing())  # Voxel spacing in mm
    origin = np.array(image.GetOrigin())  # Image origin in mm

    # Compute the center in physical space
    center = origin + 0.5 * (size - 1) * spacing
    return center

def apply_predefined_rotation(image_path, output_image_path, flip_flag):
    """
    Apply a predefined rotation transform to an image.
    """
    # Load the image
    image = sitk.ReadImage(image_path)

    # Compute image center
    # center = get_image_center(image)

    # # Create a 3D affine transform
    # rotation_transform = sitk.AffineTransform(3)

    # # Predefined 3x3 rotation matrix from the given ITK-SNAP transform
    # rotation_matrix = [0.9940078181565525, 0.10930899982914358, 3.609975987687544e-17, 
    #                    0.10445197478257746, -0.9498401752651603, 0.2947769095710655, 
    #                    -0.032221769157939166, 0.29301055272566684, 0.9555661011064231]

    # # Set rotation matrix and ensure rotation happens around the image center
    # rotation_transform.SetMatrix(rotation_matrix)
    # rotation_transform.SetFixedParameters(center)

    # # Resample the image using the rotation transform
    # resampled_image = sitk.Resample(
    #     image,
    #     image,  # Reference image for size and spacing
    #     rotation_transform,
    #     sitk.sitkLinear,  # Interpolation
    #     0,  # Default pixel value for areas outside original image
    #     image.GetPixelID()
    # )

    if flip_flag:
        resampled_image = sitk.Flip(image, (True, False, False), True)
    else:
        resampled_image = image

    # Save the rotated image
    sitk.WriteImage(resampled_image, output_image_path)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Apply predefined rotation to an image")
    parser.add_argument("image_path", type=str, help="Path to the input image (NIfTI format)")
    parser.add_argument("output_image_path", type=str, help="Path to output")
    parser.add_argument('-f', action='store_true')

    args = parser.parse_args()

    # Apply the rotation
    apply_predefined_rotation(args.image_path, args.output_image_path, args.f)