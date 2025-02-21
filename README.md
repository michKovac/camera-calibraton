# Camera Calibration and Image Preprocessing
This project focuses on image preprocessing and calibration for hyperspectral SWIR, RGB, and NIR cameras. It includes data collection and image alignment functionalities.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Image Alignment Pipeline](#image-alignment-pipeline)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to provide tools for calibrating and preprocessing images captured by hyperspectral SWIR, RGB, and NIR cameras. Proper calibration and preprocessing are essential for accurate data analysis and image alignment. The addition of the NIR camera enhances the capabilities of the project, allowing for more comprehensive image analysis and processing.

## Features
- Image preprocessing for hyperspectral SWIR, RGB, and NIR cameras
- Camera calibration routines
- Data processing tools
- Image alignment algorithms for three camera types (SWIR, RGB, NIR)

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Collection**: Use the provided scripts to collect images from all three cameras.
2. **Calibration**: Run the calibration routines to calibrate the cameras.
3. **Preprocessing**: Apply preprocessing steps to the collected images.
4. **Image Alignment**: Use the alignment algorithms to align images from all three cameras.
5. **Collection Undistortion**: Use the `undistort_collection.py` script to undistort a batch of images.
6. **Collection Alignment**: Use the `align_collection.py` script to align a batch of images.

## Image Alignment Pipeline
Follow these steps to align images from three different cameras:

0. **Preparation**: Choose at least 20 good images of a checkerboard from each camera for calibration. Ensure the images include different angles and distances.
1. **Calibration**: Calculate the camera parameter matrix for each camera using the checkerboard images.
    ```bash
    python calibration.py --path <checkerboard_images_folder> --outdir <calibration_results_path>
    ```
2. **Undistortion**: Use the calculated parameters to undistort images from all cameras.
    ```bash
    python undistort.py --path <images_folder> --outdir <undistorted_images_folder> --matrix <camera_parameters_matrix_file .pkl>
    ```
3. **Homography Calculation**: Calculate the homography matrix from one pair of images from different cameras. Choose a pair of images with good visibility of features in both images at different depths.
    ```bash
    python calculate_homography.py --swir <swir_image_path .jpg> --rgb <rgb_image_path .jpg> --nir <nir_image_path .jpg>
    ```
4. **Image Alignment**: Use the obtained homography matrix to align a batch of images from the three different cameras.
    ```bash
    python align_images.py --swir <swir_images_folder> --rgb <rgb_images_folder> --nir <nir_images_folder> --output <aligned_images_folder> --h-swir-rgb <homography_matrix_file .pkl> --h-nir-rgb <homography_matrix_file .pkl> --show 
    ```
    If you use the "--show" option, you can navigate through all images by pressing 'a' for forward and 'd' for backward. If you want to process all images without displaying them, please omit this option.

## Collection Undistortion
Use the `undistort_collection.py` script to undistort a batch of images from multiple data directories.

### Usage
```bash
python undistort_collection.py --base_path <base_dataset_path> --rgb_matrix <rgb_matrix_file> --swir_matrix <swir_matrix_file> --nir_matrix <nir_matrix_file> --start_range <start_range> --end_range <end_range>
```

### Arguments
- `--base_path`: Base path of the dataset.
- `--rgb_matrix`: Matrix file for RGB data.
- `--swir_matrix`: Matrix file for SWIR data.
- `--nir_matrix`: Matrix file for NIR data.
- `--start_range`: Start range of data directories (default: 1).
- `--end_range`: End range of data directories (default: 20).

### Data Collection Base Directory Structure
In the data collection base directory, there should be folders named `data1`, `data2`, `data3`, ..., `datax`. Each `data` folder should contain three subfolders: `rgb`, `swir`, and `nir`.

Example structure:
```
<base_path>/
    ├── data1/
    │   ├── rgb/
    │   ├── swir/
    │   └── nir/
    ├── data2/
    │   ├── rgb/
    │   ├── swir/
    │   └── nir/
    └── datax/
        ├── rgb/
        ├── swir/
        └── nir/
```
## Collection Alignment
Use the `align_collection.py` script to align a batch of images from multiple data directories.

### Usage
```bash
python align_collection.py --base_path <base_dataset_path> --h_swir_rgb <homography_matrix_file> --h_nir_rgb <homography_matrix_file> --start <start_range> --end <end_range>
```

### Arguments
- `--base_path`: Base path where all collected images are stored.
- `--h_swir_rgb`: Path to the SWIR-RGB homography matrix file.
- `--h_nir_rgb`: Path to the NIR-RGB homography matrix file (optional).
- `--start`: Start of the range for data iteration (default: 1).
- `--end`: End of the range for data iteration (default: 29).

### Data Collection Base Directory Structure
In the data collection base directory, there should be folders named `data1_und`, `data2_und`, `data3_und`, ..., `datax_und`. Each `data_und` folder should contain three subfolders: `rgb`, `swir`, and `nir`.

Example structure:
```
<base_path>/
    ├── data1_und/
    │   ├── rgb/
    │   ├── swir/
    │   └── nir/
    ├── data2_und/
    │   ├── rgb/
    │   ├── swir/
    │   └── nir/
    └── datax_und/
        ├── rgb/
        ├── swir/
        └── nir/
```
## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


