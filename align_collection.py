import os
import subprocess
from tqdm import tqdm
import argparse

def main(base_path, matrix_file_rgb, matrix_file_nir=None, start=1, end=29):
    script_path = "align_images.py"

    # Iterate over the data directories from start to end
    for i in tqdm(range(start, end + 1), desc="Processing data"):
        rgb_path = os.path.join(base_path, f"data{i}_und/rgb")
        swir_path = os.path.join(base_path, f"data{i}_und/swir")
        nir_path = os.path.join(base_path, f"data{i}_und/nir") if matrix_file_nir else None
        out_dir = os.path.join(base_path, f"data{i}_aligned")

        # Construct the command to run the script
        command = [
            "python", script_path,
            "--swir", swir_path,
            "--rgb", rgb_path,
            "--homography-rgb", matrix_file_rgb,
            "--output", out_dir,
        ]

        # Add NIR-related arguments if NIR is present
        if matrix_file_nir and os.path.exists(nir_path):
            command.extend([
                "--nir", nir_path,
                "--homography-nir", matrix_file_nir
            ])

        # Run the command
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align images using homography matrix.")
    parser.add_argument("--base_path", required=True, help="Base path where all collected images are stored.")
    parser.add_argument("--matrix_file_rgb", required=True, help="Path to the RGB-SWIR homography matrix file.")
    parser.add_argument("--matrix_file_nir", help="Path to the NIR-SWIR homography matrix file (optional).")
    parser.add_argument("--start", type=int, default=1, help="Start of the range for data iteration.")
    parser.add_argument("--end", type=int, default=29, help="End of the range for data iteration.")

    args = parser.parse_args()
    main(args.base_path, args.matrix_file_rgb, args.matrix_file_nir, args.start, args.end)


    