import cv2
import numpy as np

import glob
from robotpy_apriltag import AprilTagDetector

# Define the dimensions of the AprilTag grid
grid_size = (6, 6)  # Adjust based on your AprilTag grid size
tag_size = 1.0  # Size of one AprilTag in your grid (in same units as real-world measurements)

# Prepare object points based on real-world coordinates
objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
objp *= tag_size

# Arrays to store object points and image points from all images
objpoints = []
imgpoints = []

# Initialize AprilTag detector
detector = AprilTagDetector()
detector.addFamily("tag36h11")
config =detector.getConfig() 
# Load calibration images
images = glob.glob('april_samples/*.jpeg')
img_dimension =cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY).shape[::-1]


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect AprilTags
    results = detector.detect(gray)
    
    if results:
        imgpoints_ = []
        for r in results:
            imgpoints_.append(r.center)
        
        if len(imgpoints_) == grid_size[0] * grid_size[1]:
            objpoints.append(objp)
            imgpoints.append(np.array(imgpoints_, dtype=np.float32))
        
        # Draw detected tags
        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            
            # Draw bounding box
            cv2.line(img, ptA, ptB, (0, 255, 0), 2)
            cv2.line(img, ptB, ptC, (0, 255, 0), 2)
            cv2.line(img, ptC, ptD, (0, 255, 0), 2)
            cv2.line(img, ptD, ptA, (0, 255, 0), 2)
            
            # Draw center
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
        
        cv2.imshow('AprilTag Detection', img)
        cv2.waitKey(0)
else:
    print('No AprilTag detected in the image')

cv2.destroyAllWindows()

# Perform camera calibration
#ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_dimension, None, None)

# Save the calibration results
#np.savez('camera_params.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)