from img2dataset import download
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/pair_deduplicated/2025-18",
        help="Directory containing input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/images",
        help="Directory to save filtered datasets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)

    download(
        processes_count=16,
        thread_count=32,
        url_list=args.input_dir,
        image_size=256,
        output_folder=output_dir,
        output_format="parquet",
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        enable_wandb=True,
        number_sample_per_shard=10000,
        save_additional_columns=["page_title", "page_url", "text"],
        distributor="multiprocessing",
    )
