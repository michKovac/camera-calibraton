import os
import shutil

# Define the source and destination directories
source_dir = '/home/michal/Documents/datacollections/extracted_images'
rgb_cam_dir = '/chess_samples/rgb_cam'
swir_cam_dir = '/chess_samples/swir_cam'

# Create destination directories if they don't exist
os.makedirs(rgb_cam_dir, exist_ok=True)
os.makedirs(swir_cam_dir, exist_ok=True)

# Iterate over the files in the source directory
for filename in os.listdir(source_dir):
    if 'color' in filename:
        shutil.move(os.path.join(source_dir, filename), os.path.join(rgb_cam_dir, filename))
    elif 'swir' in filename:
        shutil.move(os.path.join(source_dir, filename), os.path.join(swir_cam_dir, filename))