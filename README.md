# camera-calibraton# Camera Calibration and Image Preprocessing
This project focuses on image preprocessing and calibration for a hyperspectral SWIR camera and an RGB camera. It includes data collection and image alignment functionalities.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Image Alignment Pipeline](#image-alignment-pipeline)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to provide tools for calibrating and preprocessing images captured by hyperspectral SWIR and RGB cameras. Proper calibration and preprocessing are essential for accurate data analysis and image alignment.

## Features
- Image preprocessing for hyperspectral SWIR and RGB cameras
- Camera calibration routines
- Data processing tools
- Image alignment algorithms

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Collection**: Use the provided scripts to collect images from both cameras.
2. **Calibration**: Run the calibration routines to calibrate the cameras.
3. **Preprocessing**: Apply preprocessing steps to the collected images.
4. **Image Alignment**: Use the alignment algorithms to align images from both cameras.

## Image Alignment Pipeline
Follow these steps to align images from two different cameras:

0. **Preparation**: Choose at least 20 good images of a checkerboard from each camera for calibration. Ensure the images include different angles and distances.
1. **Calibration**: Calculate the camera parameter matrix for each camera using the checkerboard images.
    ```bash
    python calibration.py --path <checkerboard_images_folder> --outdir <calibration_results_path>
    ```
2. **Undistortion**: Use the calculated parameters to undistort images from both cameras.
    ```bash
    python undistort.py --path <images_folder> --outdir <undistorted_images_folder> --matrix <camera_parameters_matrix_file .pkl>
    ```
3. **Homography Calculation**: Calculate the homography matrix from one pair of images from different cameras.
    ```bash
    python calculate_homography.py --swir <swir_image_path .jpg> --rgb <rgb_image_path .jpg>
    ```
4. **Image Alignment**: Use the obtained homography matrix to align a batch of images from the two different cameras.
    ```bash
    python align_images.py --swir <swir_images_folder> --rgb <rgb_images_folder> --output <aligned_images_folder> --homography <homography_matrix_file .pkl> --show 
    ```
    If you you "--show" argumend, you can iterate trough all images by pressing 'a' (forward) and 'd' (backward).

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

