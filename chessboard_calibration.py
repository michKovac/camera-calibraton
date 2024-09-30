import cv2
import numpy as np
import glob

# Define the dimensions of the checkerboard
checkerboard_size = (7, 8)
square_size = 1.0  # Adjust based on your checkerboard square size

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on real-world coordinates
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = []
imgpoints = []

# Load calibration images
images = glob.glob('chess_samples/*.jpeg')

img_dimension =cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY).shape[::-1]

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    if ret:
        objpoints.append(objp)
        
        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow('Checkerboard', cv2.resize(img, (img_dimension[0]//2, img_dimension[1]//2)))
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_dimension, None, None)

# Save the calibration results
np.savez('camera_params.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)