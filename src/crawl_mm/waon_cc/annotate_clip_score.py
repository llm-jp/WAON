import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import io
import pandas as pd
import os
from typing import List, Optional
from argparse import ArgumentParser
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F


class CustomDataset(Dataset):
    """画像バイト列をprocessorで前処理するデータセット（並列化可能）"""

    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_bytes = self.df.iloc[idx]["jpg"]
        caption = self.df.iloc[idx]["caption"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # 前処理をここで済ませる（辞書で返される）
        processed = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in processed.items()}  # バッチ次元除去


def compute_scores(
    df,
    model,
    processor,
    batch_size: int,
    num_workers: int = 4,
) -> List[Optional[float]]:
    dataset = CustomDataset(df, processor)

    def collate_fn(batch):
        # batch: list of dict of tensors
        return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # 結果を格納する配列（元の順序を保持）
    results = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        image_embeds = output.image_embeds
        text_embeds = output.text_embeds
        # コサイン類似度（CLIPスコア）
        scores = F.cosine_similarity(image_embeds, text_embeds, dim=-1).tolist()
        results.extend(scores)

    return results


def process_file(
    input_file: str,
    model,
    processor,
    batch_size: int,
    output_file: str,
    num_workers: int = 4,
):
    df = pd.read_parquet(input_file, engine="pyarrow")

    # 画像バイト列のリストを取得
    # DataLoaderを使用して処理
    scores = compute_scores(df, model, processor, batch_size, num_workers)

    # punsafeカラム追加
    df["similarity"] = scores

    # 保存
    df.to_parquet(output_file, index=False, engine="pyarrow")


def process_files(
    input_files: List[str],
    model,
    processor,
    batch_size: int,
    output_file: str,
    num_workers: int = 4,
):
    df = pd.concat(
        [pd.read_parquet(file, engine="pyarrow") for file in input_files],
        ignore_index=True,
    )
    scores = compute_scores(df, model, processor, batch_size, num_workers)
    df["similarity"] = scores
    df.to_parquet(output_file, index=False, engine="pyarrow")


def parse_args():
    parser = ArgumentParser(
        description="Compute perceptual hashes for images in parquet files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean_phash_deduplicated/2025-18",
        help="Directory containing parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean_phash_deduplicated_clip",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for processing images.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/siglip2-base-patch16-256",
        help="Model name to use for feature extraction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AutoModel.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .eval()
        .to(device)
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    num_workers = 4

    os.makedirs(output_dir, exist_ok=True)
    chunk_size = 10  # チャンクサイズを設定
    for i in range(0, 100000, chunk_size):
        files = [
            os.path.join(args.input_dir, f"{j:05d}.parquet")
            for j in range(i, i + chunk_size)
            if os.path.exists(os.path.join(args.input_dir, f"{j:05d}.parquet"))
        ]

        output_path = os.path.join(
            output_dir,
            f"{files[0].split('/')[-1].split('.')[0]}_{files[-1].split('/')[-1].split('.')[0]}.parquet",
        )
        if os.path.exists(output_path):
            print(f"[SKIP] {output_path} already exists")
            continue
        process_files(
            files, model, processor, args.batch_size, output_path, num_workers
        )


if __name__ == "__main__":
    main()
