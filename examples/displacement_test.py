import cv2
import numpy as np
import pickle

# Load images
img1 = cv2.imread('result_new/swir_calib/swir_image_000139.jpg')
img2 = cv2.imread('result_new/rgb_calib/rgb_image_000139.jpg')

def load_calibration(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            cam_matrix = data['actual_camera_matrix']
            dist_coeffs = data['dist_coeffs']
            optimal_camera_matrix = data['optimal_camera_matrix']
            return cam_matrix, dist_coeffs, optimal_camera_matrix
def save_homography(filename, matrix):
        """
        Save the homography matrix to a file.
        
        :param filename: The name of the file to save the homography matrix.
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'homography_matrix': matrix
            }, f)


# Chessboard pattern size and square size (replace with actual values)
pattern_size = (7, 8)  # e.g., 9x6 chessboard pattern
square_size = 0.05  # e.g., 5 cm per square

# Find chessboard corners
found1, corners1 = cv2.findChessboardCorners(img1, pattern_size)
found2, corners2 = cv2.findChessboardCorners(img2, pattern_size)

if not found1 or not found2:
    print("Error, cannot find the chessboard corners in both images.")
    exit()

# Calculate object points (3D points in world coordinates)
object_points = []
for i in range(pattern_size[1]):
    for j in range(pattern_size[0]):
        object_points.append([j * square_size, i * square_size, 0])
object_points = np.array(object_points, dtype=np.float32)

# Load the camera intrinsics and distortion coefficients


cam_matrix1, dist_coeffs1, optimal_camera_matrix1 = load_calibration('camera_rgb_params.pkl')
cam_matrix2, dist_coeffs2, optimal_camera_matrix2 = load_calibration('camera_swir_params.pkl')


# Solve PnP for both images (rvec1, tvec1 and rvec2, tvec2)
retval1, rvec1, tvec1 = cv2.solvePnP(object_points, corners1, cam_matrix1, dist_coeffs1)
retval2, rvec2, tvec2 = cv2.solvePnP(object_points, corners2, cam_matrix2, dist_coeffs2)

# Convert rotation vectors to rotation matrices
R1, _ = cv2.Rodrigues(rvec1)
R2, _ = cv2.Rodrigues(rvec2)

# Function to compute the relative transformation between two poses
def compute_c2mc1(R1, tvec1, R2, tvec2):
    R_1to2 = np.dot(R2, R1.T)
    tvec_1to2 = np.dot(R2, -np.dot(R1.T, tvec1)) + tvec2
    return R_1to2, tvec_1to2

# Compute relative transformation between camera 1 and camera 2
R_1to2, tvec_1to2 = compute_c2mc1(R1, tvec1, R2, tvec2)

# Normal vector in world coordinates
normal = np.array([[0], [0], [1]], dtype=np.float64)

# Compute the normal vector in camera 1's frame
normal1 = np.dot(R1, normal)

# Origin point (0, 0, 0) in world coordinates
origin = np.zeros((3, 1), dtype=np.float64)

# Compute the origin point in camera 1's frame
origin1 = np.dot(R1, origin) + tvec1

# Compute the inverse of the distance to the plane in camera 1's frame
d_inv1 = 1.0 / np.dot(normal1.T, origin1)

# Function to compute homography
def compute_homography(R_1to2, tvec_1to2, d_inv, normal):
    homography = R_1to2 + d_inv * np.dot(tvec_1to2, normal.T)
    return homography

# Compute the Euclidean homography
homography_euclidean = compute_homography(R_1to2, tvec_1to2, d_inv1, normal1)

# Compute the full homography using the camera matrix
homography = np.dot(np.dot(cam_matrix2, homography_euclidean), np.linalg.inv(cam_matrix1))

# Normalize homography matrices by dividing by the element at [2, 2]
homography /= homography[2, 2]
homography_euclidean /= homography_euclidean[2, 2]

# Print the resulting homographies
print("Homography (Euclidean):")
print(homography_euclidean)
save_homography('euc_homography.pkl', homography_euclidean)

print("Homography (with Camera Matrix):")
print(homography)
save_homography('di', homography)
