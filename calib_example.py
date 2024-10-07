import cv2
import numpy as np
import glob
from src.pair_cameras_calibration.calibration import CamCalibration
import os

    # Function to pad images to the same size
def pad_image(image, target_height, target_width):
        height, width = image.shape[:2]
        pad_height = (target_height - height) // 2
        pad_width = (target_width - width) // 2
        padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image




def do_calibration(calib, images, camera='rgb', output='save'):
    """
    Perform camera calibration and image undistortion.
    Parameters:
    calib (CamCalibration): An instance of the CamCalibration class.
    images (list): List of image file paths to be used for calibration.
    camera (str, optional): The camera type 'rgb' or 'swir' (default is 'rgb').
    output (str, optional): The output mode, either 'save' to save the images or 'show' to display them (default is 'save').
    calibrate (bool, optional): Whether to perform calibration (default is True).
    print_chessboard (bool, optional): Whether to print the chessboard pattern (default is False).
    Returns:
    None
    """
    result_path = f'result/{camera}_calib/'
    os.makedirs(result_path, exist_ok=True)
    calib.calibrate()

    for img_path in images:
        image_name = os.path.basename(img_path)
        image_name = os.path.splitext(image_name)[0]

        img = cv2.imread(img_path)
        undistorted_img = calib.undistort(img, method='undistort')
        remapped_img = calib.undistort(img, method='remap')
        print(f'image resolution {img.shape}, undistorte: {undistorted_img.shape}, remapped: {remapped_img.shape}')
   
    # Find the maximum width and height among all images
        max_height = max(img.shape[0], undistorted_img.shape[0], remapped_img.shape[0])
        max_width = max(img.shape[1], undistorted_img.shape[1], remapped_img.shape[1])
    # Pad all images to the maximum size
        img_padded = pad_image(img, max_height, max_width)
        remapped_img_padded = pad_image(remapped_img, max_height, max_width)
        undistorted_img_padded = pad_image(undistorted_img, max_height, max_width)
    # Combine padded images
    # Resize images to half their size
        img_padded_resized = cv2.resize(img_padded, (max_width // 3, max_height // 3))
        remapped_img_padded_resized = cv2.resize(remapped_img_padded, (max_width // 3, max_height // 3))
        undistorted_img_padded_resized = cv2.resize(undistorted_img_padded, (max_width // 3, max_height // 3))
    # Combine resized images
        combined_img_padded = np.hstack((img_padded_resized, remapped_img_padded_resized, undistorted_img_padded_resized))
        if output == 'show':
            cv2.imshow('Original | Undistorted | Remapped (Padded)', combined_img_padded)
            cv2.waitKey(0)
        elif output == 'save':
            cv2.imwrite(f'{result_path}{image_name}.png', remapped_img)
    cv2.destroyAllWindows()

camera = 'rgb'
images = glob.glob(f'chess_samples/{camera}_cam/*.png')
matrix_save_path = f'camera_{camera}_params'
calib = CamCalibration(f'chess_samples/{camera}_cam', savepath=matrix_save_path)
do_calibration(calib, images, output='show', camera='rgb')