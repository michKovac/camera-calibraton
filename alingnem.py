import cv2
import numpy as np

# Load camera calibration parameters (if available)
#with np.load('camera_params.npz') as data:
   # camera_matrix = data['camera_matrix']
   # dist_coeffs = data['dist_coeffs']

# Read the images
img1 = cv2.imread('img3.jpeg')
img2 = cv2.imread('img4.jpeg')
keepPercent = 0.2
# Undistort images (optional, only if calibration data is available)
#if 'camera_matrix' in locals() and 'dist_coeffs' in locals():
 #   img1 = cv2.undistort(img1, camera_matrix, dist_coeffs)
  #  img2 = cv2.undistort(img2, camera_matrix, dist_coeffs)

# Convert images to grayscale
def align_images(img1, img2, keepPercent):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

# Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography using RANSAC
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# Use homography to warp image
    height, width, channels = img1.shape
    aligned_img2 = cv2.warpPerspective(img2, h, (width, height))
    return aligned_img2

aligned_img2 = align_images(img1, img2, keepPercent)

# Show aligned images
im1_save = cv2.resize(img1, (img1.shape[1]//3, img1.shape[0]//3))
im2_save = cv2.resize(img2, (img1.shape[1]//3, img2.shape[0]//3))
aligned_img2_save = cv2.resize(aligned_img2, (aligned_img2.shape[1]//3, aligned_img2.shape[0]//3))

font = cv2.FONT_HERSHEY_TRIPLEX
org = (50, 50)
fontScale = 1
color = (0, 0, 0)
thickness = 2
 
image = cv2.putText(im1_save, 'img 1', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(im2_save, 'img 2', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
image = cv2.putText(aligned_img2_save, 'img 2 aligned to img 1', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

result = np.hstack((im1_save, im2_save, aligned_img2_save))
# Using cv2.putText() method

cv2.imwrite('aligned_img2.jpeg', result )
cv2.imshow('Aligned Image 2', result)
cv2.waitKey(0)
cv2.destroyAllWindows()