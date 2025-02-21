import cv2 as cv
import pickle
import numpy as np
import os
import tqdm

class ImageAlignment:
    def __init__(self, homography_matrix=None, homography_matrix_nir=None):
        """
        Initialize with optional homography matrices for SWIR-RGB and NIR-RGB alignment.
        
        :param homography_matrix: Optional precomputed SWIR-RGB homography matrix
        :param homography_matrix_nir: Optional precomputed NIR-RGB homography matrix
        """
        self.descriptor = cv.SIFT.create()
        self.matcher = cv.FlannBasedMatcher()
        
        if homography_matrix is not None:
            self.__load(homography_matrix)
        if homography_matrix_nir is not None:
            self.__load_nir(homography_matrix_nir)

    def __save(self, filename, is_nir=False):
        with open(filename, 'wb') as f:
            data = {
                'homography_matrix': self.homography_matrix_nir if is_nir else self.homography_matrix
            }
            pickle.dump(data, f)

    def __load(self, filename):
        """
        Load the homography matrix from a file.
        
        :param filename: The name of the file to load the homography matrix from.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.homography_matrix = data['homography_matrix']
            self.calibrated = True

    def __load_nir(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.homography_matrix_nir = data['homography_matrix']

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

    def calculate_homography(self, image_path_rgb, image_path_swir, image_path_nir=None, lowe_ratio=0.75):
        """
        Calculate homography matrices for SWIR-RGB and optionally NIR-RGB alignment.
        
        :param image_path_rgb: Path to RGB image
        :param image_path_swir: Path to SWIR image
        :param image_path_nir: Optional path to NIR image
        :return: Tuple of homography matrices (rgb-swir, nir_rgb_matrix)
        """
        # Calculate RGB-SWIR homography
        swir_rgb_matrix = self._calculate_single_homography(image_path_rgb, image_path_swir, lowe_ratio)
        self.homography_matrix = swir_rgb_matrix
        self.__save('h_swir_rgb.pkl')

        # Calculate NIR-SWIR homography if NIR image provided
        nir_rgb_matrix = None
        if image_path_nir:
            nir_rgb_matrix = self._calculate_single_homography(image_path_nir, image_path_swir, lowe_ratio)
            self.homography_matrix_nir = nir_rgb_matrix
            self.__save('h_nir_rgb.pkl', is_nir=True)

        return swir_rgb_matrix, nir_rgb_matrix

    def _calculate_single_homography(self, source_path, target_path, lowe_ratio):
        """Helper method to calculate homography between two images"""
        image_rgb = cv.imread(source_path, cv.IMREAD_COLOR)
        image_swir = cv.imread(target_path)
        image_rgb_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
        image_swir_gray = cv.imread(target_path, cv.IMREAD_GRAYSCALE)

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
            #homography_matrix, _ = cv.estimateAffine2D(pts_swir, pts_rgb)
            #homography_matrix = np.vstack((homography_matrix, [0, 0, 1]))
            homography_matrix, _ = cv.findHomography(pts_swir, pts_rgb, cv.RANSAC, 3.0)
            return homography_matrix
        return None

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
    
    def align_images(self, path_swir, path_rgb, path_nir=None, homography_mat_swir_rgb=None, homography_mat_nir_rgb=None):
        """
        Align RGB and optionally NIR images to SWIR image.
        
        :return: Tuple of (aligned_rgb, aligned_nir, original_swir) or (aligned_rgb, original_swir)
        """
        if homography_mat_swir_rgb is None:
            homography_mat_swir_rgb = self.homography_matrix

        # Load NIR homography matrix if not provided
        if homography_mat_nir_rgb is None:
            homography_mat_nir_rgb = self.homography_matrix_nir

        # Read the images
        swir_img = cv.imread(path_swir, cv.IMREAD_COLOR)
        rgb_img = cv.imread(path_rgb, cv.IMREAD_COLOR)

        # Warp the SWIR image to RGB perspective
        warped_swir = cv.warpPerspective(swir_img, homography_mat_swir_rgb, (rgb_img.shape[1], rgb_img.shape[0]))
        # Convert to BGR if necessary (assuming SWIR is in grayscale)
        warped_swir = cv.cvtColor(warped_swir, cv.COLOR_BGR2GRAY)
        warped_swir = cv.cvtColor(warped_swir, cv.COLOR_GRAY2BGR)

        if path_nir and homography_mat_nir_rgb is not None:
            nir_img = cv.imread(path_nir, cv.IMREAD_COLOR)
            # Warp the NIR image to RGB perspective
            nir_img = cv.imread(path_nir, cv.IMREAD_COLOR)
            nir_img = cv.resize(nir_img, (rgb_img.shape[1], rgb_img.shape[0]))
            warped_nir = cv.warpPerspective(nir_img, homography_mat_nir_rgb, (rgb_img.shape[1], rgb_img.shape[0]), flags=cv.INTER_LINEAR)
            # Convert to BGR if necessary
            warped_nir = cv.cvtColor(warped_nir, cv.COLOR_BGR2GRAY)
            warped_nir = cv.cvtColor(warped_nir, cv.COLOR_GRAY2BGR)

            # Crop the images to their intersection
            #cropped_warped_swir, cropped_im_rgb = self.__crop_to_intersection(warped_swir, rgb_img)
            #cropped_warped_nir, _ = self.__crop_to_intersection(warped_nir, rgb_img)
            cropped_warped_swir, cropped_warped_nir, cropped_im_rgb = self.__crop_to_intersection_three(warped_swir, warped_nir, rgb_img)


            return cropped_im_rgb, cropped_warped_nir, cropped_warped_swir  # Return aligned images

        # If NIR is not provided, just return the cropped SWIR and RGB images
        cropped_warped_swir, cropped_im_rgb = self.__crop_to_intersection(warped_swir, rgb_img)
        return cropped_im_rgb, cropped_warped_swir

    def __crop_to_intersection(self, warped_swir, im_rgb):
        """
        Crop the warped SWIR image and the RGB image to their intersection without black borders.
        
        :param warped_swir: The warped SWIR image.
        :param im_rgb: The RGB image.
        :return: The cropped warped SWIR image and the cropped RGB image.
        """
        # Find the first non-black pixel from each side for the warped SWIR image
        top = self.__find_non_black_edge(warped_swir, axis=0)
        bottom = self.__find_non_black_edge(warped_swir, axis=0, reverse=True)
        left = self.__find_non_black_edge(warped_swir, axis=1)
        right = self.__find_non_black_edge(warped_swir, axis=1, reverse=True)

        # Crop the images using the found edges
        cropped_warped_swir = warped_swir[top:bottom, left:right]
        cropped_im_rgb = im_rgb[top:bottom, left:right]

        return cropped_warped_swir, cropped_im_rgb

    def __crop_to_intersection_three(self, warped_swir, warped_nir, rgb):
        """Crop three images to their intersection"""
        # Find the non-black regions in both warped images
        top = max(
            self.__find_non_black_edge(warped_swir, axis=0),
            self.__find_non_black_edge(warped_nir, axis=0)
        )
        bottom = min(
            self.__find_non_black_edge(warped_swir, axis=0, reverse=True),
            self.__find_non_black_edge(warped_nir, axis=0, reverse=True)
        )
        left = max(
            self.__find_non_black_edge(warped_swir, axis=1),
            self.__find_non_black_edge(warped_nir, axis=1)
        )
        right = min(
            self.__find_non_black_edge(warped_swir, axis=1, reverse=True),
            self.__find_non_black_edge(warped_nir, axis=1, reverse=True)
        )

        return (
            warped_swir[top:bottom, left:right],
            warped_nir[top:bottom, left:right],
            rgb[top:bottom, left:right]
        )
        
    def update_opacity(self, rgb_alpha, swir_alpha, nir_alpha, rgb_alin, swir_alin, nir_alin):
        # Resize the images to the same dimensions
        height = max(rgb_alin.shape[0], swir_alin.shape[0], nir_alin.shape[0])
        width = max(rgb_alin.shape[1], swir_alin.shape[1], nir_alin.shape[1])
        
        rgb_alin_resized = cv.resize(rgb_alin, (width, height))
        swir_alin_resized = cv.resize(swir_alin, (width, height))
        nir_alin_resized = cv.resize(nir_alin, (width, height))
        
        # Normalize alpha values to be between 0 and 1
        rgb_alpha = rgb_alpha / 100.0
        swir_alpha = swir_alpha / 100.0
        nir_alpha = nir_alpha / 100.0
        
        # Blend the resized images
        blended = cv.addWeighted(rgb_alin_resized, rgb_alpha, swir_alin_resized, swir_alpha, 0)
        blended = cv.addWeighted(blended, 1.0, nir_alin_resized, nir_alpha, 0)  # Blend NIR image
        
        # Ensure the blended image is in the correct format
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        cv.imshow('Blended', blended)
        
    def align_batch(self, swir_path, rgb_path, nir_path=None, output_path="aligned", homography_mat=None, show=False):
        # Initialize opacity values
        rgb_alpha = 100
        swir_alpha = 100
        nir_alpha = 100

        if homography_mat is None:
            homography_mat = self.homography_matrix

        # Create output directories if they don't exist
        swir_alin_output_path = os.path.join(output_path, 'swir')
        rgb_alin_output_path = os.path.join(output_path, 'rgb')
        nir_alin_output_path = os.path.join(output_path, 'nir') if nir_path else None
        os.makedirs(swir_alin_output_path, exist_ok=True)
        os.makedirs(rgb_alin_output_path, exist_ok=True)
        if nir_alin_output_path:
            os.makedirs(nir_alin_output_path, exist_ok=True)

        # List all images in the directories
        swir_images = sorted([f for f in os.listdir(swir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        rgb_images = sorted([f for f in os.listdir(rgb_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        nir_images = sorted([f for f in os.listdir(nir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]) if nir_path else []

        # Ensure both directories have the same number of images
        if len(swir_images) != len(rgb_images):
            raise ValueError("The number of images in SWIR and RGB directories do not match.")
        if nir_path and len(nir_images) != len(rgb_images):
            raise ValueError("The number of images in NIR and RGB directories do not match.")

        if show:
            cv.namedWindow('Blended')
            cv.createTrackbar('RGB Opacity', 'Blended', rgb_alpha, 100, lambda x: self.update_opacity(x / 100, swir_alpha / 100, nir_alpha / 100, rgb_alin, swir_alin, nir_alin))
            cv.createTrackbar('SWIR Opacity', 'Blended', swir_alpha, 100, lambda x: self.update_opacity(rgb_alpha / 100, x / 100, nir_alpha / 100, rgb_alin, swir_alin, nir_alin))
            cv.createTrackbar('NIR Opacity', 'Blended', nir_alpha, 100, lambda x: self.update_opacity(rgb_alpha / 100, swir_alpha / 100, x / 100, rgb_alin, swir_alin, nir_alin))

        index = 0
        print('Starting image alignment...')
        pbar = tqdm.tqdm(total=len(swir_images), desc='Processing images')
        while True:
            swir_image = swir_images[index]
            rgb_image = rgb_images[index]

            swir_image_path = os.path.join(swir_path, swir_image)
            rgb_image_path = os.path.join(rgb_path, rgb_image)

            # Align SWIR and RGB images
            swir_alin, rgb_alin = self.align_images(swir_image_path, rgb_image_path)

            if nir_path:
                nir_image = nir_images[index]
                nir_image_path = os.path.join(nir_path, nir_image)
                #nir_alin, _ = self.align_images(nir_image_path, rgb_image_path)  # Align NIR to RGB
                rgb_alin, nir_alin, swir_alin = self.align_images(swir_image_path, rgb_image_path, nir_image_path)
                nir_aligned_output_file = os.path.join(nir_alin_output_path, nir_image)
                cv.imwrite(nir_aligned_output_file, nir_alin)
                
            swir_aligned_output_file = os.path.join(swir_alin_output_path, swir_image)
            rgb_alin_output_file = os.path.join(rgb_alin_output_path, rgb_image)
            cv.imwrite(swir_aligned_output_file, swir_alin)
            cv.imwrite(rgb_alin_output_file, rgb_alin)

            pbar.update(1)

            # Update opacity with NIR image
            if show:
                opac_rgb = cv.getTrackbarPos('RGB Opacity', 'Blended') 
                opac_swir = cv.getTrackbarPos('SWIR Opacity', 'Blended') 
                opac_nir = cv.getTrackbarPos('NIR Opacity', 'Blended') 
                self.update_opacity(opac_rgb, opac_swir, opac_nir, rgb_alin, swir_alin, nir_alin)  # Initialize with the first call

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
                    opac_rgb = cv.getTrackbarPos('RGB Opacity', 'Blended')
                    opac_swir = cv.getTrackbarPos('SWIR Opacity', 'Blended')
                    opac_nir = cv.getTrackbarPos('NIR Opacity', 'Blended')
                    self.update_opacity(opac_rgb, opac_swir, opac_nir, rgb_alin, swir_alin, nir_alin)  # Update opacity on trackbar change
            else:
                index += 1
                if index >= len(swir_images):
                    break
        pbar.close()
        print('Image alignment complete.')
        print(f'Total number of processed images: {index}')
        print(f'Aligned images saved to {output_path}')
            