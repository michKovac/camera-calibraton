import cv2 as cv
import numpy as np
import os
from src.pair_cameras_calibration.image_aligement import ImageAlignment



swir_path = '/home/michal/git/camera-calibraton/result_latest/data12_0.005/swir_calib'
rgb_path = '/home/michal/git/camera-calibraton/result_latest/data12_0.005/rgb_calib'
swir_alin_output_path = '/home/michal/git/camera-calibraton/result_latest/data12_0.005/swir_ali'
rgb_alin_output_path = '/home/michal/git/camera-calibraton/result_latest/data12_0.005/rgb_ali'


# Call the function with the provided paths
matrix = 'homography_matrix.pkl'
#matrix = 'homography_matrix.pkl'
alingment = ImageAlignment(matrix)
alingment.align_batch(swir_path, rgb_path, swir_alin_output_path, rgb_alin_output_path, show=True)

