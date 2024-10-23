from src.pair_cameras_calibration.image_aligement import ImageAlignment

swir_path = '/home/michal/Documents/datacollections/1019/undistordted_swir/data2'
rgb_path = '/home/michal/Documents/datacollections/1019/undistordted_rgb/data2'
out_path = '/home/michal/Documents/datacollections/1019/aligned/data2'


# Call the function with the provided paths
matrix = 'homography_1019_000786.pkl'
#matrix = 'homography_matrix.pkl'
alingment = ImageAlignment(matrix)
alingment.align_batch(swir_path, rgb_path, out_path, show=True)

