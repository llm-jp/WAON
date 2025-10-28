import json
import trafilatura
from argparse import ArgumentParser
import os
from tqdm import tqdm
from crawl_mm.utils.ja_classifier import is_japanese
from concurrent.futures import ProcessPoolExecutor
from loguru import logger


def process_html(input_path, output_path):
    results = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for i, line in tqdm(enumerate(infile), desc="Processing HTML", unit="line"):
            data = json.loads(line)
            html = data["html"]
            url = data["url"]
            text = trafilatura.extract(
                html,
                favor_precision=True,
                include_comments=False,
                include_tables=False,
                deduplicate=True,
            )
            if not text:
                continue
            try:
                metadata = trafilatura.extract_metadata(html)
            except Exception as e:
                logger.error(f"Metadata extraction error: {e}")
                continue
            title = metadata.title
            if not title:
                continue
            if not is_japanese(text):
                continue

            write_data = {
                "title": title,
                "url": url,
                "text": text,
                "html": html,
            }
            results.append(write_data)
    # save as parquet
    import polars as pl

    df = pl.DataFrame(results)
    df.write_parquet(output_path)

    # with open(output_path, "w", encoding="utf-8") as outfile:
    #     for result in results:
    #         outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Processed {len(results)} lines from {input_path} to {output_path}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/html/2025-18",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/goodhtml",
        help="Output directory for good HTML files",
    )
    parser.add_argument(
        "--max_num_files",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.output_dir, os.path.basename(args.input_dir))
    os.makedirs(output_dir, exist_ok=True)
    input_paths = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".jsonl")
    ]
    input_paths = sorted(input_paths)[: args.max_num_files]

    with ProcessPoolExecutor() as executor:
        futures = []
        for input_path in input_paths:
            output_path = os.path.join(
                output_dir, os.path.basename(input_path).replace(".jsonl", ".parquet")
            )
            if os.path.exists(output_path) and not args.overwrite:
                logger.info(f"File already exists: {output_path}")
                continue
            futures.append(executor.submit(process_html, input_path, output_path))
        for future in tqdm(futures):
            future.result()
    logger.info(f"Processed {len(futures)} files.")
