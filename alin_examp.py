from src.pair_cameras_calibration.image_aligement import ImageAlignment

swir_path = '/media/share/Dataset/swir_dataset/collect/1128_aftersnow/data20_und/swir/'
rgb_path = '/media/share/Dataset/swir_dataset/collect/1128_aftersnow/data20_und/rgb/'
out_path = '/home/michal/Documents/RASMD/1128_aftersnowy/data20_aligned'


# Call the function with the provided paths
matrix = 'homography_1128_001420.pkl'
#matrix = 'homography_matrix.pkl'
alingment = ImageAlignment(matrix)
alingment.align_batch(swir_path, rgb_path, out_path, show=True)

