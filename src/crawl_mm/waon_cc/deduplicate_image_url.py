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
        default="data/cc/pair/2025-18",
        help="Directory containing input JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/pair_deduplicated",
        help="Directory to save output Parquet files.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Number of files to process before logging progress.",
    )

    return parser.parse_args()


def url_to_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)

    logging_steps = args.logging_steps
    use_wandb = False
    if use_wandb:
        wandb.init(
            project="waon_cc",
            name="deduplicate_image_url",
            config={
                "input_dir": input_dir,
                "output_dir": output_dir,
                "max_elements": 1_000_000_000,
                "error_rate": 0.01,
            },
        )

    input_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".parquet")
        ]
    )

    # Estimate: 100M URLs, 1% false positive rate
    seen_hashes = BloomFilter(max_elements=1_000_000_000, error_rate=0.01)
    seen_caption_hashes = BloomFilter(max_elements=1_000_000_000, error_rate=0.01)

    processed_file_num = 0
    for input_file in tqdm(input_files, desc="Processing files"):
        df = pd.read_parquet(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
        keep_indices = []

        for idx, image_url, caption in zip(df.index, df["url"], df["caption"]):
            image_url_hash = url_to_hash(image_url)
            caption_hash = url_to_hash(caption)
            if image_url_hash in seen_hashes:
                continue
            if caption_hash in seen_caption_hashes:
                continue
            seen_hashes.add(image_url_hash)
            seen_caption_hashes.add(caption_hash)
            keep_indices.append(idx)

        df_filtered = df.loc[keep_indices]
        print(f"Filtered down to {len(df_filtered)} rows from {len(df)}")
        output_file = os.path.join(
            output_dir, os.path.basename(input_file).replace(".jsonl", ".parquet")
        )
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
