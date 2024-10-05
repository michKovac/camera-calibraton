import cv2 as cv
import numpy as np

# read the images
ir = cv.imread('paired/frame21_modified2_swir.jpg', cv.IMREAD_GRAYSCALE)
ir_bgr = cv.imread('paired/frame21_modified2_swir.jpg')
rgb = cv.imread('paired/frame21_rgb.jpg', cv.IMREAD_COLOR)


descriptor = cv.SIFT.create()
matcher = cv.FlannBasedMatcher()

# get features from images
kps_ir, desc_ir = descriptor.detectAndCompute(ir, mask=None)
gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
kps_color, desc_color = descriptor.detectAndCompute(gray, mask=None)

# find the corresponding point pairs
if (desc_ir is not None and desc_color is not None and len(desc_ir) >=2 and len(desc_color) >= 2):
    rawMatch = matcher.knnMatch(desc_color, desc_ir, k=2)
matches = []
# ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
ratio = 0.75
for m in rawMatch:
    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
        matches.append((m[0].trainIdx, m[0].queryIdx))

# convert keypoints to points
pts_ir, pts_color = [], []
for id_ir, id_color in matches:
    pts_ir.append(kps_ir[id_ir].pt)
    pts_color.append(kps_color[id_color].pt)
pts_ir = np.array(pts_ir, dtype=np.float32)
pts_color = np.array(pts_color, dtype=np.float32)

# compute homography
if len(matches) > 4:
    #H, status = cv.findHomography(pts_ir, pts_color, cv.RANSAC)
    H, _ = cv.estimateAffine2D(pts_ir, pts_color)
    H = np.vstack((H, [0, 0, 1]))

warped = cv.warpPerspective(ir, H, (rgb.shape[1], rgb.shape[0]))
warped = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)

# visualize the result
# Show aligned images
im1_save = cv.resize(rgb, (rgb.shape[1]//3, rgb.shape[0]//3))
im2_save = cv.resize(ir_bgr, (ir_bgr.shape[1]//3, ir_bgr.shape[0]//3))
aligned_img2_save = cv.resize(warped, (warped.shape[1]//3, warped.shape[0]//3))

font = cv.FONT_HERSHEY_TRIPLEX
org = (50, 50)
fontScale = 1
color = (255, 255, 255)
thickness = 2
 
image = cv.putText(im1_save, 'RGB', org, font, 
                   fontScale, color, thickness, cv.LINE_AA)
image = cv.putText(im2_save, 'IR', org, font, 
                   fontScale, color, thickness, cv.LINE_AA)
image = cv.putText(aligned_img2_save, 'aligned', org, font, 
                   fontScale, color, thickness, cv.LINE_AA)

result = np.hstack((im1_save, im2_save, aligned_img2_save))
# Using cv2.putText() method

cv.imwrite('aligned_img2.jpeg', result )
cv.imshow('Aligned Image 2', result)
cv.waitKey(0)
cv.destroyAllWindows()