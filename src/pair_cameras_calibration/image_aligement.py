import cv2 as cv
import pickle
import numpy as np

class ImageAlignment:
    def __init__(self, image_path_rgb, image_path_swir, homography_matrix=None):

        self.image_path_rgb = image_path_rgb
        self.image_path_swir = image_path_swir
        self.descriptor = cv.SIFT.create()
        self.matcher = cv.FlannBasedMatcher()
        if homography_matrix is not None:
            self.__load('homography_matrix.pkl')
        else:
            self.homography_matrix = homography_matrix
        if self.image_path_rgb is not None and image_path_swir is not None:
            self.image_rgb = cv.imread(self.image_path_rgb,cv.IMREAD_COLOR)
            self.image_swir = cv.imread(self.image_path_swir)
            self.image_rgb_gray = cv.cvtColor(self.image_rgb, cv.COLOR_BGR2GRAY)
            self.image_swir_gray = cv.imread(self.image_path_swir, cv.IMREAD_GRAYSCALE)
        else:
            raise ValueError('Image paths are not provided')
            
    
    def __save(self, filename):
            with open(filename, 'wb') as f:
                pickle.dump({
                    'homography_matrix': self.homography_matrix
                }, f)

    
    def __load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.homography_matrix = data['homography_matrix']
            self.calibrated = True

    def calculate_homography(self, lowe_ratio=0.75):
        kps_swir, desc_swir = self.descriptor.detectAndCompute(self.image_swir_gray, mask=None)
        kps_rgb, desc_rgb = self.descriptor.detectAndCompute(self.image_rgb_gray, mask=None)
        
        # find the corresponding point pairs  
        if (desc_swir is not None and desc_rgb is not None and len(desc_swir) >=2 and len(desc_rgb) >= 2):
            rawMatch = self.matcher.knnMatch(desc_rgb, desc_swir, k=2)
        matches = []
        # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
        
        for m in rawMatch:
            if len(m) == 2 and m[0].distance < m[1].distance * lowe_ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # convert keypoints to points
        pts_swir, pts_rgb = [], []
        for id_swir, id_rgb in matches:
            pts_swir.append(kps_swir[id_swir].pt)
            pts_rgb.append(kps_rgb[id_rgb].pt)    
        pts_swir = np.array(pts_swir, dtype=np.float32)
        pts_rgb = np.array(pts_rgb, dtype=np.float32)
        # compute homography
        if len(matches) > 4:
            #H, status = cv.findHomography(pts_ir, pts_color, cv.RANSAC)
            self.homography_matrix, _ = cv.estimateAffine2D(pts_swir, pts_rgb)
            self.homography_matrix = np.vstack((self.homography_matrix, [0, 0, 1]))
            self.__save('homography_matrix.pkl')
        return self.homography_matrix
    
    def align_images(self, im_swir_grey=None, im_rgb=None, homography_mat=None):
        if im_swir_grey is None:
            im_swir_grey = self.image_swir_gray
        if im_rgb is None:
            im_rgb = self.image_rgb
        if homography_mat is None:
            homography_mat = self.homography_matrix
        if self.homography_matrix is not None:
            warped_swir = cv.warpPerspective(im_swir_grey, homography_mat, (im_rgb.shape[1], im_rgb.shape[0]))
            warped_swir = cv.cvtColor(warped_swir, cv.COLOR_GRAY2BGR)
            return warped_swir, im_rgb
        else:
            raise ValueError('Homography matrix is not calculated')
        
   
       
    
        