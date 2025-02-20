from src.pair_cameras_calibration.image_aligement import ImageAlignment
import argparse

def main():
    parser = argparse.ArgumentParser(description='Align images using homography matrix.')
    parser.add_argument('--swir', type=str, required=True, help='Path to the SWIR calibration images')
    parser.add_argument('--rgb', type=str, required=True, help='Path to the RGB calibration images')
    parser.add_argument('--nir', type=str, help='Path to the NIR calibration images (optional)')
    parser.add_argument('--output', type=str, required=True, help='Output path for aligned images')
    parser.add_argument('--homography-rgb', type=str, required=True, help='Path to the RGB-SWIR homography matrix')
    parser.add_argument('--homography-nir', type=str, help='Path to the NIR-SWIR homography matrix')
    parser.add_argument('--show', action='store_true', help='Show the alignment process')

    args = parser.parse_args()

    alignment = ImageAlignment(
        args.homography_rgb,
        args.homography_nir if args.nir else None
    )
    alignment.align_batch(
        args.swir,
        args.rgb,
        args.output,
        nir_path=args.nir,
        show=args.show
    )

if __name__ == '__main__':
    main()