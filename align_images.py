from src.pair_cameras_calibration.image_aligement import ImageAlignment
import argparse

def main():
    parser = argparse.ArgumentParser(description='Align images using homography matrix.')
    parser.add_argument('--swir', type=str, required=True, help='Path to the SWIR undistorted images')
    parser.add_argument('--rgb', type=str, required=True, help='Path to the RGB undistorted images')
    parser.add_argument('--nir', type=str, help='Path to the NIR undistorted images (optional)')
    parser.add_argument('--output', type=str, required=True, help='Output path for aligned images')
    parser.add_argument('--h_swir_rgb', type=str, required=True, help='Path to the SWIR-RGB homography matrix')
    parser.add_argument('--h_nir_rgb', type=str, help='Path to the NIR-RGB homography matrix')
    parser.add_argument('--show', action='store_true', help='Show the alignment process')

    args = parser.parse_args()

    alignment = ImageAlignment(
        args.h_swir_rgb,
        args.h_nir_rgb if args.nir else None
    )
    alignment.align_batch(
        args.swir,
        args.rgb,
        nir_path=args.nir,  # Ensure this is passed correctly
        output_path=args.output,
        show=args.show
    )

if __name__ == "__main__":
    main()