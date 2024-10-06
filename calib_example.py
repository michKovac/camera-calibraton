import cv2
import numpy as np
import glob
from calibration import CamCalibration

calib = CamCalibration('chess_samples/rgb_cam', print_chessboard=False, matrix_file_path='camera_params.pkl')
#calib.calibrate()

images = glob.glob('chess_samples/rgb_cam/*.png')

for img_path in images:
    img = cv2.imread(img_path)
    undistorted_img = calib.undistort(img, method='undistort')
    remapped_img = calib.undistort(img, method='remap')
    # Add padding to undistorted_img and remapped_img to match the original image size
    undistorted_img_padded = cv2.copyMakeBorder(
        undistorted_img,
        top=0,
        bottom=img.shape[0] - undistorted_img.shape[0],
        left=0,
        right=img.shape[1] - undistorted_img.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    remapped_img_padded = cv2.copyMakeBorder(
        remapped_img,
        top=0,
        bottom=img.shape[0] - remapped_img.shape[0],
        left=0,
        right=img.shape[1] - remapped_img.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    combined_img = np.hstack((img, undistorted_img_padded, remapped_img_padded))

    cv2.imshow('Original | Undistorted | Remapped', combined_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()