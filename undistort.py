import cv2
import glob
from src.pair_cameras_calibration.calibration import CamCalibration
import os
import argparse

def do_calibration(calib, images, result_path):
    os.makedirs(result_path, exist_ok=True)

    for img_path in images:
        image_name = os.path.basename(img_path)
        image_name = os.path.splitext(image_name)[0]
        img = cv2.imread(img_path)
        remapped_img = calib.undistort(img, method='remap')
        cv2.imwrite(f'{result_path}/{image_name}.jpg', remapped_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Undistort images using camera calibration.')
    parser.add_argument('--path', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--outdir', type=str, default='undistorted', help='Path to the directory to save undistorted images.')
    parser.add_argument('--matrix', type=str, required=True, help='Path to the camera matrix file.')

    args = parser.parse_args()

    images_path = args.path
    result_path = args.outdir
    matrix_file_path = args.matrix

    images = glob.glob(f'{images_path}/*.jpg') + glob.glob(f'{images_path}/*.png')

    calib = CamCalibration(images_path, matrix_file_path=matrix_file_path)
    do_calibration(calib, images, result_path)