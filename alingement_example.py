import cv2 as cv
import numpy as np
from src.pair_cameras_calibration.image_aligement import ImageAlignment

ir_path = 'paired/calib_swir_camera_image_raw_1728031532351120217.png'
rgb_path = 'paired/calib_camera_image_color_1728031532559420138.png'
image_alignment = ImageAlignment(rgb_path, ir_path, homography_matrix='homography_matrix.pkl')
#image_alignment.calculate_homography(lowe_ratio=0.65)
aligned, rgb = image_alignment.align_images()


cropped_img1_resized = cv.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))
cropped_warped_img2_resized = cv.resize(aligned, (aligned.shape[1] // 2, aligned.shape[0] // 2))

cv.imshow('Aligned', np.hstack((cropped_img1_resized, cropped_warped_img2_resized)))
cv.waitKey(0)
cv.destroyAllWindows()