import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image = cv2.imread('result_data1/swir_ali/swir_image_000000.jpg')

# Step 2: Convert the image to grayscale (assuming the rhombus is a different color than the black background)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Load the imag

# Step 3: Threshold the image to separate the rhombus from the black background
_, thresh_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

# Step 4: Find the bounding box for the non-black part of the image (the rhombus)
coords = np.column_stack(np.where(thresh_image > 0))
x, y, w, h = cv2.boundingRect(coords)

# Step 5: Compute the diagonals of the rhombus from the bounding box
# The diagonals of the rhombus are the width and height of the bounding box
diag1 = w  # One diagonal (horizontal diagonal of bounding box)
diag2 = h  # The other diagonal (vertical diagonal of bounding box)

# Step 6: The largest inner rectangle is sqrt(2)/2 of the diagonals of the rhombus
# The largest rectangle will have sides that are sqrt(2)/2 times the lengths of the diagonals
inner_w = int(diag1 / np.sqrt(2))
inner_h = int(diag2 / np.sqrt(2))

# Step 7: Center the largest rectangle inside the rhombus
# Calculate the top-left corner of the inner rectangle
inner_x = x + (w - inner_w) // 2
inner_y = y + (h - inner_h) // 2

# Step 8: Crop the largest inner rectangle
cropped_image = image[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]

cv2.imshow('Original',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


cv2.imshow('Cropped',cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)