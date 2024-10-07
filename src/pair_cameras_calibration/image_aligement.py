import cv2 as cv
import pickle
import numpy as np

class ImageAlignment:
    def __init__(self, image_path_rgb, image_path_swir, homography_matrix=None):
        """
        Initialize the ImageAlignment class with paths to RGB and SWIR images and an optional homography matrix.
        
        :param image_path_rgb: Path to the RGB image.
        :param image_path_swir: Path to the SWIR image.
        :param homography_matrix: Optional precomputed homography matrix.
        """
        self.image_path_rgb = image_path_rgb
        self.image_path_swir = image_path_swir
        self.descriptor = cv.SIFT.create()
        self.matcher = cv.FlannBasedMatcher()
        
        # Load homography matrix if provided
        if homography_matrix is not None:
            self.__load('homography_matrix.pkl')
        else:
            self.homography_matrix = homography_matrix
        
        # Load and process images
        if self.image_path_rgb is not None and image_path_swir is not None:
            self.image_rgb = cv.imread(self.image_path_rgb, cv.IMREAD_COLOR)
            self.image_swir = cv.imread(self.image_path_swir)
            self.image_rgb_gray = cv.cvtColor(self.image_rgb, cv.COLOR_BGR2GRAY)
            self.image_swir_gray = cv.imread(self.image_path_swir, cv.IMREAD_GRAYSCALE)
        else:
            raise ValueError('Image paths are not provided')
    
    def __save(self, filename):
        """
        Save the homography matrix to a file.
        
        :param filename: The name of the file to save the homography matrix.
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'homography_matrix': self.homography_matrix
            }, f)

    def __load(self, filename):
        """
        Load the homography matrix from a file.
        
        :param filename: The name of the file to load the homography matrix from.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.homography_matrix = data['homography_matrix']
            self.calibrated = True
    def __find_non_black_edge(self,img, axis, reverse=False):
        if reverse:
            img = np.flip(img, axis=axis)
        for i in range(img.shape[axis]):
            if axis == 0:
                if np.any(img[i, :] > 0):
                    return img.shape[axis] - i if reverse else i
            else:
                if np.any(img[:, i] > 0):
                    return img.shape[axis] - i if reverse else i
        return 0
    
    def __crop_black_borders(self,image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Find the first non-black pixel from each side
        top = self.__find_non_black_edge(gray, axis=0)
        bottom = self.__find_non_black_edge(gray, axis=0, reverse=True)
        left = self.__find_non_black_edge(gray, axis=1)
        right = self.__find_non_black_edge(gray, axis=1, reverse=True)

        # Crop the image using the found edges
        return image[top:bottom, left:right]
    
    def __crop_image_to_dimension(self, image, side, dimension):
                """
                Crop the image from the chosen side to the specified dimension.
                
                :param image: The image to be cropped.
                :param side: The side from which to crop ('left', 'right', 'top', 'bottom').
                :param dimension: The dimension to crop to.
                :return: The cropped image.
                """
                if side == 'left':
                    return image[:, :dimension]
                elif side == 'right':
                    return image[:, -dimension:]
                elif side == 'top':
                    return image[:dimension, :]
                elif side == 'bottom':
                    return image[-dimension:, :]
                else:
                    raise ValueError("Side must be one of 'left', 'right', 'top', 'bottom'")

    def calculate_homography(self, lowe_ratio=0.75):
        """
        Calculate the homography matrix using feature matching.
        
        :param lowe_ratio: The ratio for Lowe's ratio test to filter matches.
        :return: The computed homography matrix.
        """
        kps_swir, desc_swir = self.descriptor.detectAndCompute(self.image_swir_gray, mask=None)
        kps_rgb, desc_rgb = self.descriptor.detectAndCompute(self.image_rgb_gray, mask=None)
        
        # Find the corresponding point pairs
        if desc_swir is not None and desc_rgb is not None and len(desc_swir) >= 2 and len(desc_rgb) >= 2:
            rawMatch = self.matcher.knnMatch(desc_rgb, desc_swir, k=2)
        matches = []
        
        # Apply Lowe's ratio test to filter matches
        for m in rawMatch:
            if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
        # Convert keypoints to points
        pts_swir, pts_rgb = [], []
        for id_swir, id_rgb in matches:
            pts_swir.append(kps_swir[id_swir].pt)
            pts_rgb.append(kps_rgb[id_rgb].pt)
        pts_swir = np.array(pts_swir, dtype=np.float32)
        pts_rgb = np.array(pts_rgb, dtype=np.float32)
        
        # Compute homography if enough matches are found
        if len(matches) > 4:
            self.homography_matrix, _ = cv.estimateAffine2D(pts_swir, pts_rgb)
            self.homography_matrix = np.vstack((self.homography_matrix, [0, 0, 1]))
            self.__save('homography_matrix.pkl')
        return self.homography_matrix
    
    def align_images(self, im_swir_grey=None, im_rgb=None, homography_mat=None):
        """
        Align the SWIR image to the RGB image using the homography matrix.
        
        :param im_swir_grey: Optional SWIR image in grayscale.
        :param im_rgb: Optional RGB image.
        :param homography_mat: Optional homography matrix.
        :return: The aligned SWIR image and the original RGB image.
        """
        if im_swir_grey is None:
            im_swir_grey = self.image_swir_gray
        if im_rgb is None:
            im_rgb = self.image_rgb
        if homography_mat is None:
            homography_mat = self.homography_matrix
        
        if self.homography_matrix is not None:
            warped_swir = cv.warpPerspective(im_swir_grey, homography_mat, (im_rgb.shape[1], im_rgb.shape[0]))
            warped_swir = cv.cvtColor(warped_swir, cv.COLOR_GRAY2BGR)
            warped_swir = self.__crop_black_borders(warped_swir)
            h, w = warped_swir.shape[:2]
            croped_rgb = self.__crop_image_to_dimension(im_rgb, 'right', w)
            croped_rgb = self.__crop_image_to_dimension(croped_rgb, 'bottom', h)
            return warped_swir, croped_rgb
        else:
            raise ValueError('Homography matrix is not calculated')
        