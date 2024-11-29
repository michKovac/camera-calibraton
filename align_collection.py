import os
import subprocess
from tqdm import tqdm
import argparse

def main(base_path, matrix_file, start, end):
    script_path = "align_images.py"

    # Iterate over the data directories from start to end
    for i in tqdm(range(start, end + 1), desc="Processing RGB data"):
        rgb_path = os.path.join(base_path, f"data{i}_und/rgb")
        swir_path = os.path.join(base_path, f"data{i}_und/swir")
        out_dir = os.path.join(base_path, f"data{i}_aligned")

        # Construct the command to run the script for RGB data
        rgb_command = [
            "python", script_path,
            "--swir", swir_path,
            "--rgb", rgb_path,
            "--homography", matrix_file,
            "--output", out_dir,
        ]
        # Run the command for RGB data
        subprocess.run(rgb_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align images using homography matrix.")
    parser.add_argument("--base_path", required=True, help="Base path where all collected images are stored.")
    parser.add_argument("--matrix_file", required=True, help="Path to the homography matrix file.")
    parser.add_argument("--start", type=int, default=1, help="Start of the range for data iteration.")
    parser.add_argument("--end", type=int, default=29, help="End of the range for data iteration.")

    args = parser.parse_args()
    main(args.base_path, args.matrix_file, args.start, args.end)


    