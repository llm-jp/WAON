import os
import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser


def compute_phash_from_bytes(img_bytes):
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def parse_args():
    parser = ArgumentParser(
        description="Compute perceptual hashes for images in parquet files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean/2025-18",
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean_phash",
        help="Directory to save parquet files with phash.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes to use.",
    )
    return parser.parse_args()


def process_file(path, output_path):
    df = pd.read_parquet(path)
    phashes = [compute_phash_from_bytes(b) for b in df["jpg"]]
    df["phash"] = phashes
    df = df[df["phash"].notnull()]
    if df.empty:
        print(f"[SKIP] No valid images in {path}")
        return
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)

    file_paths = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".parquet")
    ]
    file_paths.sort()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_file, path, output_dir) for path in file_paths
        ]
        for input_path in file_paths:
            output_path = os.path.join(output_dir, os.path.basename(input_path))
            if os.path.exists(output_path):
                print(f"[SKIP] {output_path} already exists")
                continue
            executor.submit(process_file, input_path, output_path)
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            pass

    print(f"Processed {len(file_paths)} files. Output saved to {output_dir}.")
