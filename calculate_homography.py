import cv2 as cv
import numpy as np
import argparse
from src.pair_cameras_calibration.image_aligement import ImageAlignment

def main(ir_path, rgb_path, nir_path=None):
    image_alignment = ImageAlignment()
    hm_rgb, hm_nir = image_alignment.calculate_homography(rgb_path, ir_path, nir_path, lowe_ratio=0.75)
    print(f'RGB-SWIR homography matrix: {hm_rgb}')
    if hm_nir is not None:
        print(f'NIR-SWIR homography matrix: {hm_nir}')

    if nir_path:
        aligned_rgb, aligned_nir, swir = image_alignment.align_images(ir_path, rgb_path, nir_path)
        # Display three images side by side
        resized_rgb = cv.resize(aligned_rgb, (aligned_rgb.shape[1] // 3, aligned_rgb.shape[0] // 3))
        resized_nir = cv.resize(aligned_nir, (aligned_nir.shape[1] // 3, aligned_nir.shape[0] // 3))
        resized_swir = cv.resize(swir, (swir.shape[1] // 3, swir.shape[0] // 3))
        cv.imshow('Aligned Images', np.hstack((resized_rgb, resized_nir, resized_swir)))
    else:
        aligned_rgb, swir = image_alignment.align_images(ir_path, rgb_path)
        resized_rgb = cv.resize(aligned_rgb, (aligned_rgb.shape[1] // 3, aligned_rgb.shape[0] // 3))
        resized_swir = cv.resize(swir, (swir.shape[1] // 3, swir.shape[0] // 3))
        cv.imshow('Aligned Images', np.hstack((resized_rgb, resized_swir)))

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate homography and align images.')
    parser.add_argument('--swir', type=str, required=True, help='Path to the SWIR image')
    parser.add_argument('--rgb', type=str, required=True, help='Path to the RGB image')
    parser.add_argument('--nir', type=str, help='Path to the NIR image (optional)')
    args = parser.parse_args()

    main(args.swir, args.rgb, args.nir)