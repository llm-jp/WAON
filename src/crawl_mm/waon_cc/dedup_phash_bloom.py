from tqdm import tqdm
import hashlib
import os
import pandas as pd
from bloom_filter2 import BloomFilter
import wandb
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Deduplicate image URLs in JSONL files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean_phash/2025-18",
        help="Directory containing input JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean_phash_deduplicated",
        help="Directory to save output Parquet files.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Number of files to process before logging progress.",
    )
    parser.add_argument(
        "--max_elements",
        type=int,
        default=50_000_000,
        help="Maximum number of elements for the Bloom filter.",
    )
    parser.add_argument(
        "--error_rate",
        type=float,
        default=0.01,
        help="Error rate for the Bloom filter.",
    )

    return parser.parse_args()


def url_to_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, os.path.basename(input_dir))
    logging_steps = args.logging_steps
    use_wandb = False
    if use_wandb:
        wandb.init(
            project="waon_cc",
            name="deduplicate_phash",
        )

    input_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".parquet")
        ]
    )
    os.makedirs(output_dir, exist_ok=True)

    # Estimate: 100M URLs, 1% false positive rate
    seen_hashes = BloomFilter(
        max_elements=args.max_elements, error_rate=args.error_rate
    )

    processed_file_num = 0
    for input_file in tqdm(input_files, desc="Processing files"):
        df = pd.read_parquet(input_file, engine="pyarrow")
        keep_indices = []
        phashes = df["phash"].tolist()

        for idx, phash in zip(df.index, phashes):
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)
            keep_indices.append(idx)

        df_filtered = df.loc[keep_indices]
        print(f"Filtered down to {len(df_filtered)} rows from {len(df)}")
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        try:
            df_filtered.to_parquet(output_file, index=False, engine="pyarrow")
        except Exception as e:
            print(f"Error writing {output_file}: {e}")
        processed_file_num += 1
        if processed_file_num % logging_steps == 0:
            print(f"Processed {processed_file_num} files so far.")
            if use_wandb:
                wandb.log(
                    {
                        "processed_file_num": processed_file_num,
                    }
                )
