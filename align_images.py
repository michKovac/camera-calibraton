from src.pair_cameras_calibration.image_aligement import ImageAlignment
import argparse

def main():
    parser = argparse.ArgumentParser(description='Align images using homography matrix.')
    parser.add_argument('--swir', type=str, required=True, help='Path to the SWIR calibration images.')
    parser.add_argument('--rgb', type=str, required=True, help='Path to the RGB calibration images.')
    parser.add_argument('--output', type=str, required=True, help='Output path for aligned  and RGB images.')
    parser.add_argument('--homography', type=str, default='homography_matrix_best.pkl', help='Path to the homography matrix file.')
    parser.add_argument('--show', action='store_true', help='Show the alignment process.')

    args = parser.parse_args()

    alingment = ImageAlignment(args.homography)
    alingment.align_batch(args.swir, args.rgb, args.output, show=args.show)

if __name__ == '__main__':
    main()