import re
import json
import chardet
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
import os
from argparse import ArgumentParser
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
import requests


# lang="ja" 検出
LANG_JA_REGEX = re.compile(r'<html[^>]*lang=["\']?ja["\']?', re.IGNORECASE)


def is_japanese_html_raw(html_str):
    return LANG_JA_REGEX.search(html_str)


def try_decode(raw_html):
    try:
        return raw_html.decode("utf-8")
    except UnicodeDecodeError:
        result = chardet.detect(raw_html[:2048])
        encoding = result["encoding"] or "utf-8"
        try:
            return raw_html.decode(encoding)
        except Exception:
            return None


def process_warc(input_path, output_path):
    results = []
    with requests.get(input_path, stream=True) as r:
        r.raise_for_status()
        for record in ArchiveIterator(r.raw):
            if record.rec_type != "response":
                continue

            content_type = record.http_headers.get_header("Content-Type")
            if not content_type or "html" not in content_type:
                continue

            url = record.rec_headers.get_header("WARC-Target-URI")
            html = record.content_stream().read()

            html = try_decode(html)
            if html is None:
                continue

            if not is_japanese_html_raw(html):
                continue

            write_data = {
                "url": url,
                "html": html,
            }
            results.append(write_data)
    with open(output_path, "w", encoding="utf-8") as outfile:
        for write_data in results:
            outfile.write(json.dumps(write_data, ensure_ascii=False) + "\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/cc/url/2025-18.txt",
        help="File containing urls of warc files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cc/html",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--max_num_files",
        type=int,
        default=None,
        help="Maximum number of WARC files to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(
        args.output_dir, os.path.basename(args.input_file).split(".")[0]
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(args.input_file, "r") as f:
        input_paths = f.readlines()
    input_paths = [url.strip() for url in input_paths]
    warc_urls = sorted(input_paths)
    input_paths = input_paths[: args.max_num_files]
    with ProcessPoolExecutor() as executor:
        futures = []
        for input_path in input_paths:
            output_path = os.path.join(
                output_dir,
                os.path.basename(input_path).replace(".warc.gz", ".jsonl"),
            )
            if os.path.exists(output_path) and not args.overwrite:
                logger.info(f"File already exists: {output_path}")
                continue
            futures.append(executor.submit(process_warc, input_path, output_path))
        for future in tqdm(futures):
            future.result()

    logger.info("All files processed.")
