import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
from validate_image import count_unique_colors


def validate_image_bytes(row):
    try:
        original_width = row["original_width"]
        original_height = row["original_height"]
        if original_width < 150 or original_height < 150:
            return False
        aspect_ratio = original_width / original_height
        if aspect_ratio > 2 or aspect_ratio < 0.5:
            return False
        img = Image.open(io.BytesIO(row["jpg"])).convert("RGB")
        if count_unique_colors(img) < 32:
            return False
        return True
    except Exception:
        return False  # 壊れた画像などに対応


def process_file(input_path, output_path):
    try:
        df = pd.read_parquet(input_path)
        df = df[df["status"] == "success"].reset_index(drop=True)
        records = df.to_dict("records")

        valid_records = [record for record in records if validate_image_bytes(record)]
        df_valid = pd.DataFrame(valid_records)

        if not df_valid.empty:
            df_valid.to_parquet(output_path, index=False, engine="pyarrow")
    except Exception as e:
        print(f"Error processing {path}: {e}")


def parse_args():
    parser = ArgumentParser(
        description="Compute perceptual hashes for images in parquet files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/images/2025-18",
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/images_size-clean",
        help="Output directory.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of parallel workers."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)

    input_paths = sorted(
        [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".parquet")
        ]
    )

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for input_path in input_paths:
            output_path = os.path.join(output_dir, os.path.basename(input_path))
            if os.path.exists(output_path):
                print(f"Skipping existing output: {output_path}")
                continue
            futures.append(executor.submit(process_file, input_path, output_path))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
