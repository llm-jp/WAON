from bs4 import BeautifulSoup
from urllib.parse import urljoin
from argparse import ArgumentParser
import os
from tqdm import tqdm
from crawl_mm.waon_wiki.wiki2pair import is_valid_caption, is_valid_image_url
import ray
from loguru import logger


def extract_image_caption_pairs(html, base_url=""):
    soup = BeautifulSoup(html, "html.parser")
    pairs = []

    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        image_url = urljoin(base_url, src)

        # 1. <figure><figcaption> 優先
        caption = None
        figure = img.find_parent("figure")
        if figure:
            figcaption = figure.find("figcaption")
            if figcaption:
                caption = figcaption.get_text(strip=True)

        # 2. 次に alt 属性を使用（figcaption がなければ）
        if not caption:
            alt = img.get("alt", "").strip()
            if alt:
                caption = alt

        # caption が何かあれば追加
        if caption:
            pairs.append({"image_url": image_url, "caption": caption})

    return pairs


import polars as pl


@ray.remote
def process_html(input_path, output_path):
    try:
        df = pl.read_parquet(input_path)
    except Exception as e:
        logger.error(f"Failed to read {input_path}: {e}")
        return
    results = []

    for row in tqdm(
        df.iter_rows(named=True),
        total=df.height,
        desc=f"Processing {os.path.basename(input_path)}",
    ):
        title = row["title"]
        url = row["url"]
        text = row["text"]
        html = row["html"]

        try:
            pairs = extract_image_caption_pairs(html, url)
        except Exception as e:
            logger.warning(f"[FAIL] {url}: {e}")
            continue
        for pair in pairs:
            image_url = pair["image_url"]
            caption = pair["caption"]
            if not is_valid_image_url(image_url):
                continue
            if not is_valid_caption(caption):
                continue
            results.append(
                {
                    "url": image_url,
                    "caption": caption,
                    "page_title": title,
                    "page_url": url,
                    "text": text,
                }
            )

    if results:
        pl.DataFrame(results).write_parquet(output_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/goodhtml/2025-18",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/pair",
        help="Output directory for good HTML files",
    )
    parser.add_argument(
        "--max_num_files",
        type=int,
        default=None,
        help="Maximum number of WARC files to process",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ray.init(num_cpus=args.num_workers)

    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)

    input_paths = sorted(
        [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".parquet")
        ]
    )[: args.max_num_files]

    futures = []
    for input_path in input_paths:
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        if os.path.exists(output_path) and not args.overwrite:
            continue
        futures.append(process_html.remote(input_path, output_path))

    for _ in tqdm(ray.get(futures), desc="Processing HTML", unit="file"):
        pass

    ray.shutdown()
