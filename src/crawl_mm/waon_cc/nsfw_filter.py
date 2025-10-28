from crawl_mm.utils.nsfw import NSFWModel

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import io
import pandas as pd
import os
from typing import List, Optional
from argparse import ArgumentParser


class ImageDataset(Dataset):
    """画像バイト列を扱うPyTorchデータセット"""

    def __init__(self, df, pre_processing):
        self.df = df
        self.pre_processing = pre_processing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_bytes = self.df.iloc[idx]["jpg"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = self.pre_processing(image)
        return image


def compute_scores(
    df,
    model: NSFWModel,
    batch_size: int,
    num_workers: int = 4,
) -> List[Optional[float]]:
    """DataLoaderを使用して画像を効率的に処理"""

    dataset = ImageDataset(df, model.pre_processing)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    # 結果を格納する配列（元の順序を保持）
    results = []

    for batch_images in tqdm(dataloader, desc="Processing batches"):
        # 有効な画像のみを処理
        scores = model.predict(batch_images.to(model.device))
        results.extend(scores)

    return results


def process_batch_files(
    input_files: List[str],
    model: NSFWModel,
    batch_size: int,
    output_dir: str,
    num_workers: int = 4,
):
    dfs = []
    for input_file in input_files:
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping.")
            continue
        try:
            df = pd.read_parquet(input_file, engine="pyarrow")
            df["__file__"] = input_file  # 元のファイル名を記録
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {input_file}: {e}")

    if not dfs:
        return

    merged_df = pd.concat(dfs, ignore_index=True)

    scores = compute_scores(merged_df, model, batch_size, num_workers)
    merged_df["punsafe"] = scores
    merged_df = merged_df[merged_df["punsafe"] < 0.1]

    # 元ファイルごとに分割して保存
    for input_file, df_group in merged_df.groupby("__file__"):
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        df_group.drop(columns="__file__").to_parquet(
            output_file, index=False, engine="pyarrow"
        )


def parse_args():
    parser = ArgumentParser(
        description="Compute perceptual hashes for images in parquet files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/images_size-clean/2025-18",
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean",
        help="Directory to save parquet files with phash.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for processing images.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for DataLoader.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NSFWModel()
    input_dir = args.input_dir
    output_dir = os.path.join(args.output_dir, os.path.basename(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    batch_size = args.batch_size
    num_workers = args.num_workers
    batch_file_count = 100  # 100ファイルずつまとめて処理
    input_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".parquet")
        ]
    )
    input_files = [
        f
        for f in input_files
        if not os.path.exists(os.path.join(output_dir, os.path.basename(f)))
    ]
    if not input_files:
        print("All files already processed. Exiting.")
        return
    for i in range(0, len(input_files), batch_file_count):
        batch_files = input_files[i : i + batch_file_count]
        process_batch_files(batch_files, model, batch_size, output_dir, num_workers)


if __name__ == "__main__":
    main()
