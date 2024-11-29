import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_sift_matches(img1_path, img2_path, output_path, target_size=(800, 600), line_width=1, marker_size=8):
    """
    Display and save SIFT feature matches between two images with adjustable visualization parameters.
    
    Args:
        img1_path (str): Path to the first image
        img2_path (str): Path to the second image
        output_path (str): Path where to save the output visualization
        target_size (tuple): Target size for both images (width, height)
        line_width (int): Width of the matching lines
        marker_size (int): Size of the keypoint markers
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Resize images to the same size
    img1 = cv2.resize(img1, target_size)
    img2 = cv2.resize(img2, target_size)
    
    # Convert to RGB (matplotlib expects RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1_rgb, None)
    kp2, des2 = sift.detectAndCompute(img2_rgb, None)
    
    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Create figure and adjust size
    plt.figure(figsize=(20, 10))
    
    # Draw matches
    matched_img = cv2.drawMatches(
        img1_rgb, kp1,
        img2_rgb, kp2,
        good_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Plot with custom parameters
    plt.imshow(matched_img)
    
    # Draw keypoints and lines with custom sizes
    for match in good_matches:
        # Get the matching keypoints for each of the images
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        
        # Draw the keypoints
        plt.plot(x1, y1, 'ro', markersize=marker_size)
        plt.plot(x2 + target_size[0], y2, 'ro', markersize=marker_size)
        
        # Draw the matching line
        plt.plot([x1, x2 + target_size[0]], [y1, y2], 'g-', linewidth=line_width)
    
    plt.axis('off')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    print(f"Number of good matches: {len(good_matches)}")

if __name__ == "__main__":
    # Example usage
    img1_path = "/home/michal/Documents/RASMD/alingment_figure/undistorded_rgb_image_002245.jpg"
    img2_path = "/home/michal/Documents/RASMD/alingment_figure/undistorded_swir_image_002245.jpg"
    output_path = "/home/michal/Documents/RASMD/alingment_figure/sift_matches.png"
    
    # Call the function with custom line width and marker size
    show_sift_matches(
        img1_path, 
        img2_path,
        output_path,
        target_size=(800, 600),  # Specify desired size for both images
        line_width=1, 
        marker_size=10
    )
