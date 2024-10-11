#!/usr/bin/env python3

import argparse
from src.pair_cameras_calibration.calibration import CamCalibration

def main(image_dir, savepath, checkerboard_size, square_size):
    calib = CamCalibration(image_dir=image_dir, savepath=savepath, checkerboard_size=checkerboard_size, square_size=square_size)
    calib.calibrate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration Script")
    parser.add_argument('--path', type=str, required=True, help='Directory containing calibration images')
    parser.add_argument('--outdir', type=str, required=True, help='Path to save calibration results')
    parser.add_argument('--checkerboard_size', type=int, nargs=2, default=(7, 8), help='Checkerboard size as two integers (default: (7, 8))')
    parser.add_argument('--square_size', type=float, default=2, help='Size of a square in your defined unit (default: 2)')

    args = parser.parse_args()
    main(args.path, args.outdir, tuple(args.checkerboard_size), args.square_size)