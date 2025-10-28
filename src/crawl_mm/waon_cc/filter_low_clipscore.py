# Filter examples of which clip score is below 0.1
import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Filter images based on clip score.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean_phash_deduplicated_clip/2025-18",
        help="Directory containing parquet files with clip scores.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean_phash_deduplicated_clip_filtered",
        help="Directory to save filtered parquet files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Get all parquet files in the input directory
    input_dir = args.input_dir
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ]

    for file in tqdm(files, desc="Processing files"):
        output_file = os.path.join(output_dir, os.path.basename(file))
        if os.path.exists(output_file):
            print(f"Skipping {output_file} as it already exists.")
            continue

        # Read the parquet file
        df = pd.read_parquet(file)

        # Filter rows where 'similarity' is below 0.1
        filtered_df = df[df["similarity"] >= 0.1]

        # Save the filtered DataFrame to a new parquet file
        filtered_df.to_parquet(output_file, index=False)
