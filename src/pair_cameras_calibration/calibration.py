import cv2
import numpy as np
import glob
import pickle
from tqdm import tqdm


class CamCalibration:
    def __init__(self, image_dir, savepath='camera_params', checkerboard_size =(7,8), square_size=2.0, print_chessboard=False, image_size=None, matrix_file_path=None):
        self.image_dir = image_dir
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.print_results = print_chessboard
        self.optimal_camera_matrix = None
        self.actual_camera_matrix = None
        self.dist_coeffs = None
        self.image_roi = None
        self.savepath = savepath
        if matrix_file_path is not None:
            self.__load(matrix_file_path)
            self.calibrated = True
        else:
            self.calibrated = False
        if image_size is None:
            images = glob.glob(self.image_dir + '/*.png')
            if len(images) == 0:
                raise ValueError("No images found in the specified directory.")
            img = cv2.imread(images[0])
            self.image_size = img.shape[:2]
            self.image_size = (self.image_size[1], self.image_size[0])
        else:
            self.image_size = image_size

    def __find_obj_points(self,im_dir, check_size, sq_size):
        # Termination criteria for corner sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 56, 0.0001)

        # Prepare object points based on real-world coordinates
        objp = np.zeros((check_size[0] * check_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:check_size[0], 0:check_size[1]].T.reshape(-1, 2)
        objp *= sq_size

        # Arrays to store object points and image points from all images
        objpoints = []
        imgpoints = []

        # Load calibration images
        images = glob.glob(im_dir+'/*.png')

        img_dimension =cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY).shape[::-1]
        succ_found = 0
        print("Finding object points...")
        for fname in tqdm(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            # Find the checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, check_size, None)
    
            if ret:
                objpoints.append(objp)
                succ_found = succ_found+1
                # Refine corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                if self.print_results:
                # Draw and display the corners
                    cv2.drawChessboardCorners(img, check_size, corners2, ret)
                    cv2.imshow(f'Checkerboard {succ_found}', cv2.resize(img, (img.shape[0]//2, img.shape[1]//2)))
                    cv2.waitKey(0)
        
        if self.print_results:
            cv2.destroyAllWindows()
        print(f"Object founed: {len(objpoints)}")
        return objpoints, imgpoints
    
    def calibrate(self, im_dir=None, check_size=None, sq_size=None, im_size=None):
        if im_dir is None:
            im_dir = self.image_dir
        if check_size is None:
            check_size = self.checkerboard_size
        if sq_size is None:
            sq_size = self.square_size
        if im_size is None:
            im_size = self.image_size
        if self.calibrated:
            print("Camera already calibrated.")
            return self.actual_camera_matrix, self.dist_coeffs, self.optimal_camera_matrix
        # Perform camera calibration        
        objpoints, imgpoints = self.__find_obj_points(im_dir, check_size, sq_size)
        # Perform camera calibration
        print("Calibrating camera...")
        ret, self.actual_camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, im_size, None, None)
        self.optimal_camera_matrix, self.image_roi = cv2.getOptimalNewCameraMatrix(self.actual_camera_matrix, self.dist_coeffs, im_size, 0, im_size)
        self.calibrated = True
        print("Saving calibration results...")
        self.__save(f'{self.savepath}.pkl')
        print("Calibration complete.")
        return self.actual_camera_matrix, self.dist_coeffs, self.optimal_camera_matrix


    def undistort(self, img, camera_matrix=None, dist_c=None, new_camera_matrix=None, im_roi=None, method='undistort'):
        if camera_matrix is None:
            camera_matrix = self.actual_camera_matrix
        if dist_c is None:
            dist_c =self.dist_coeffs
        if new_camera_matrix is None:
            new_camera_matrix = self.optimal_camera_matrix
        if im_roi is None:
            im_roi = self.image_roi
        
        if self.calibrated:
            if method == 'undistort':
                result = cv2.undistort(img, camera_matrix, dist_c, None, new_camera_matrix) 
                return self.__cut_roi(result, im_roi)
            elif method == 'remap':
                mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_c, None, new_camera_matrix, self.image_size, cv2.CV_32FC1)
                result = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                return self.__cut_roi(result, im_roi)
            else:
                raise ValueError("Invalid method. Use 'undistort' or 'remap'.")
        else:
            return img
        
    def __cut_roi(self, img, roi):
        x, y, w, h = roi
        return img[y:y+h, x:x+w]

    def __save(self, filename):
        if self.calibrated:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'actual_camera_matrix': self.actual_camera_matrix,
                    'dist_coeffs': self.dist_coeffs,
                    'optimal_camera_matrix': self.optimal_camera_matrix,
                    'image_dir': self.image_dir,
                    'checkerboard_size': self.checkerboard_size,
                    'square_size': self.square_size,
                    'image_size': self.image_size,
                    "image_roi": self.image_roi
                }, f)

    def __load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.actual_camera_matrix = data['actual_camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.optimal_camera_matrix = data['optimal_camera_matrix']
            self.image_dir = data['image_dir']
            self.checkerboard_size = data['checkerboard_size']  
            self.square_size = data['square_size']
            self.image_size = data['image_size']
            self.image_roi = data['image_roi']
            # Validate loaded data
            self.calibrated = True

