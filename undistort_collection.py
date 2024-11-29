import os
import subprocess
import argparse
from tqdm import tqdm

def main(base_path, rgb_matrix_file, matrix_file, start_range, end_range):
    script_path = "undistort.py"

    # Iterate over the data directories for RGB data
    for i in tqdm(range(start_range, end_range + 1), desc="Processing RGB data"):
        input_path_rgb = os.path.join(base_path, f"data{i}/rgb")
        rgb_output_dir = os.path.join(base_path, f"data{i}_und/rgb")

        # Construct the command to run the script for RGB data
        rgb_command = [
            "python", script_path,
            "--path", input_path_rgb,
            "--outdir", rgb_output_dir,
            "--matrix", rgb_matrix_file
        ]
        print(rgb_command)
        # Run the command for RGB data
        subprocess.run(rgb_command)

    # Iterate over the data directories for SWIR data
    for i in tqdm(range(start_range, end_range + 1), desc="Processing SWIR data"):
        input_path_swir = os.path.join(base_path, f"data{i}/swir")
        output_dir = os.path.join(base_path, f"data{i}_und/swir")

        # Construct the command to run the script for SWIR data
        command = [
            "python", script_path,
            "--path", input_path_swir,
            "--outdir", output_dir,
            "--matrix", matrix_file
        ]
        print(command)
        # Run the command
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run undistortion script on dataset.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path of the dataset")
    parser.add_argument("--rgb_matrix", type=str, required=True, help="Matrix file for RGB data")
    parser.add_argument("--swir_matrix", type=str, required=True, help="Matrix file for SWIR data")
    parser.add_argument("--start_range", type=int, default=1, help="Start range of data directories")
    parser.add_argument("--end_range", type=int, default=20, help="End range of data directories")

    args = parser.parse_args()
    main(args.base_path, args.rgb_matrix, args.swir_matrix, args.start_range, args.end_range)
    
