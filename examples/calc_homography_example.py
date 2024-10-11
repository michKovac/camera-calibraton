import cv2 as cv
import numpy as np
from src.pair_cameras_calibration.image_aligement import ImageAlignment

#ir_path = 'result_latest_sel/swir_calib/swir_image_000025_1.jpg'
#rgb_path = 'result_latest_sel/rgb_calib/rgb_image_000025_1.jpg'
ir_path = '/home/michal/git/camera-calibraton/result_latest/data1/swir_calib/swir_image_000000.jpg'
rgb_path = '/home/michal/git/camera-calibraton/result_latest/data1/rgb_calib/rgb_image_000000.jpg'
image_alignment = ImageAlignment()
hm = image_alignment.calculate_homography(rgb_path, ir_path,lowe_ratio=0.75)
#hm = image_alignment.calculate_homography_chess()
print(f'homography matrix: {hm}')
aligned, rgb = image_alignment.align_images(ir_path, rgb_path)


cropped_img1_resized = cv.resize(rgb, (rgb.shape[1] // 3, rgb.shape[0] // 3))
cropped_warped_img2_resized = cv.resize(aligned, (aligned.shape[1] // 3, aligned.shape[0] // 3))
cv.imwrite("aligned.png", aligned)
cv.imshow('Aligned', np.hstack((cropped_img1_resized, cropped_warped_img2_resized)))
print(f'Aligned image shape: {aligned.shape}\n RGB image shape: {rgb.shape}')
cv.waitKey(0)
#cv.destroyAllWindows()

def update_opacity(x):
    alpha = x / 100
    beta = 1 - alpha
    blended = cv.addWeighted(cropped_img1_resized, alpha, cropped_warped_img2_resized, beta, 0)
    cv.imshow('Blended', blended)

cv.namedWindow('Blended')
cv.createTrackbar('Opacity', 'Blended', 0, 100, update_opacity)
update_opacity(0)  # Initialize with the first call

cv.waitKey(0) 
cv.destroyAllWindows()