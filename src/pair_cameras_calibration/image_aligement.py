import cv2 as cv
import pickle
import numpy as np
import os

class ImageAlignment:
    def __init__(self, homography_matrix=None):
        """
        Initialize the ImageAlignment class with paths to RGB and SWIR images and an optional homography matrix.
        
        :param image_path_rgb: Path to the RGB image.
        :param image_path_swir: Path to the SWIR image.
        :param homography_matrix: Optional precomputed homography matrix.
        """
        self.descriptor = cv.SIFT.create()
        self.matcher = cv.FlannBasedMatcher()
        
        # Load homography matrix if provided
        if homography_matrix is not None:
            self.__load(homography_matrix)
        else:
            print('need to calcutale homography')
    
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

    def calculate_homography(self,image_path_rgb, image_path_swir, lowe_ratio=0.75):
        """
        Calculate the homography matrix using feature matching.
        
        :param lowe_ratio: The ratio for Lowe's ratio test to filter matches.
        :return: The computed homography matrix.
        """
        image_rgb = cv.imread(image_path_rgb, cv.IMREAD_COLOR)
        image_swir = cv.imread(image_path_swir)
        image_rgb_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
        image_swir_gray = cv.imread(image_path_swir, cv.IMREAD_GRAYSCALE)


        kps_swir, desc_swir = self.descriptor.detectAndCompute(image_swir_gray, mask=None)
        kps_rgb, desc_rgb = self.descriptor.detectAndCompute(image_rgb_gray, mask=None)
        
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
    
    def calculate_homography_chess(self, image_path_rgb, image_path_swir):
        """
        Calculate the homography matrix using feature matching.
        
        :param lowe_ratio: The ratio for Lowe's ratio test to filter matches.
        :return: The computed homography matrix.
        """
        image_swir_gray = cv.imread(image_path_swir, cv.IMREAD_GRAYSCALE)
        image_rgb_gray = cv.imread(image_path_rgb, cv.IMREAD_GRAYSCALE)
        ret1, corners_rgb = cv.findChessboardCorners(image_rgb_gray, (7,8), None)
        ret2, corners_swir = cv.findChessboardCorners(image_swir_gray, (7,8), None)
        if ret1 and ret2:
            self.homography_matrix, _ = cv.findHomography(corners_swir, corners_rgb, cv.RANSAC, 5.0)
        
        self.__save('chess_homography_matrix.pkl')
        return self.homography_matrix
    
    def align_images(self, path_swir_grey, path_im_rgb, homography_mat=None):
        """
        Align the SWIR image to the RGB image using the homography matrix.
        
        :param im_swir_grey: Optional SWIR image in grayscale.
        :param im_rgb: Optional RGB image.
        :param homography_mat: Optional homography matrix.
        :return: The aligned SWIR image and the original RGB image.
        """
        if homography_mat is None:
            homography_mat = self.homography_matrix
        im_rgb = cv.imread(path_im_rgb, cv.IMREAD_COLOR)
        im_swir_grey = cv.imread(path_swir_grey, cv.IMREAD_GRAYSCALE)
        if self.homography_matrix is not None:
            #print(f'homography matrix: {homography_mat}')
            warped_swir = cv.warpPerspective(im_swir_grey, homography_mat, (im_rgb.shape[1], im_rgb.shape[0]))
            warped_swir = cv.cvtColor(warped_swir, cv.COLOR_GRAY2BGR)
            #warped_swir = self.__crop_black_borders(warped_swir)
            # Resize the warped SWIR image by half
            #h, w = warped_swir.shape[:2]
            #croped_rgb = self.__crop_image_to_dimension(im_rgb, 'right', w)
            #croped_rgb = self.__crop_image_to_dimension(croped_rgb, 'bottom', h)
            return warped_swir, im_rgb
        else:
            raise ValueError('Homography matrix is not calculated')
        
    def update_opacity(self,x, rgb_alin, swir_alin):
        alpha = x / 100
        beta = 1 - alpha
        # Resize the images by half
        rgb_alin_resized = cv.resize(rgb_alin, (rgb_alin.shape[1] // 2, rgb_alin.shape[0] // 2))
        swir_alin_resized = cv.resize(swir_alin, (swir_alin.shape[1] // 2, swir_alin.shape[0] // 2))
        
        # Blend the resized images
        blended = cv.addWeighted(rgb_alin_resized, alpha, swir_alin_resized, beta, 0)
        cv.imshow('Blended', blended)
        
    def align_batch(self, swir_path, rgb_path, swir_alin_output_path, rgb_alin_output_path, homography_mat=None, show=True):
        if homography_mat is None:
            homography_mat = self.homography_matrix
        # Create output directories if they don't exist
        os.makedirs(swir_alin_output_path, exist_ok=True)
        os.makedirs(rgb_alin_output_path, exist_ok=True)

        # List all images in the directories
        swir_images = sorted([f for f in os.listdir(swir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        rgb_images = sorted([f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Ensure both directories have the same number of images
        if len(swir_images) != len(rgb_images):
            raise ValueError("The number of images in SWIR and RGB directories do not match.")

        if  show:
            cv.namedWindow('Blended')
            cv.createTrackbar('Opacity', 'Blended', 0, 100, lambda x: self.update_opacity(x, rgb_alin, swir_alin))

        index = 0
        
        while True:
            swir_image = swir_images[index]
            rgb_image = rgb_images[index]

            swir_image_path = os.path.join(swir_path, swir_image)
            rgb_image_path = os.path.join(rgb_path, rgb_image)

            
            swir_alin, rgb_alin = self.align_images(swir_image_path, rgb_image_path)

            # Save the aligned and rgb images
            swir_aligned_output_file = os.path.join(swir_alin_output_path, swir_image)
            rgb_alin_output_file = os.path.join(rgb_alin_output_path, rgb_image)

            cv.imwrite(swir_aligned_output_file, swir_alin)
            cv.imwrite(rgb_alin_output_file, rgb_alin)

            if show:
                opac = cv.getTrackbarPos('Opacity', 'Blended') 
                self.update_opacity(opac, rgb_alin, swir_alin)  # Initialize with the first call

                while True:
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('a'):  # a to go forward
                        index = (index + 1) % len(swir_images)
                        break
                    elif key == ord('d'):  # d to go backward
                        index = (index - 1) % len(swir_images)
                        break
                    elif key == ord('q'):  # 'q' key to quit
                        cv.destroyAllWindows()
                        exit()
                    opac = cv.getTrackbarPos('Opacity', 'Blended')
                    self.update_opacity(opac, rgb_alin, swir_alin)  # Update opacity on trackbar change
            