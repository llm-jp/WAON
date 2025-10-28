from urllib.parse import urljoin
import requests
import json
import os
import zlib
from datetime import datetime
from argparse import ArgumentParser
from loguru import logger


def get_common_crawl_snapshot_index(index_prefix):
    index_url = urljoin(index_prefix, "collinfo.json")
    index_response = requests.get(index_url)
    return json.loads(index_response.content)


def get_main_warc_paths(
    snapshot_index, start_snapshot, end_snapshot, prefix="https://data.commoncrawl.org"
):
    beg_year, beg_week = list(map(int, start_snapshot.split("-")))
    end_year, end_week = list(map(int, end_snapshot.split("-")))
    start_date = datetime.fromisocalendar(beg_year, beg_week, 1)
    end_date = datetime.fromisocalendar(end_year, end_week, 1)

    if start_date > end_date:
        raise ValueError(
            f"Start snapshot '{start_snapshot}' is after end snapshot '{end_snapshot}'"
        )

    if beg_year < 2013 or end_year < 2013:
        logger.info("Warning: Only snapshots after 2013 are supported by this script")

    total_prefix = urljoin(prefix, "crawl-data/CC-MAIN")

    snapshot_warc_paths = {}
    for snapshot in snapshot_index:
        date = list(map(int, snapshot["id"].split("-")[2:]))

        if len(date) == 2:
            year, week = date
        else:
            continue

        if year >= 2013:
            curr_date = datetime.fromisocalendar(year, week, 1)
            if start_date <= curr_date <= end_date:
                snapshot_id = f"{year}-{week:02d}"
                warc_path = f"{total_prefix}-{snapshot_id}/warc.paths.gz"
                snapshot_warc_paths[snapshot_id] = warc_path
    return snapshot_warc_paths


def fetch_urls_from_warc_path(warc_path: str, data_domain_prefix: str) -> list[str]:
    try:
        response = requests.get(warc_path.rstrip(), stream=True)
        response.raise_for_status()
        data = zlib.decompress(response.content, zlib.MAX_WBITS | 32)
        urls = []
        for warc in data.decode("utf-8").split("\n"):
            if warc:
                urls.append(urljoin(data_domain_prefix, warc))
        return urls
    except Exception as e:
        logger.warning(f"Could not get URLs for {warc_path}: {e}")
        return []


def get_common_crawl_urls_per_snapshot(
    starting_snapshot: str,
    ending_snapshot: str,
    output_dir: str,
    data_domain_prefix="https://data.commoncrawl.org",
    index_prefix="https://index.commoncrawl.org",
):
    index = get_common_crawl_snapshot_index(index_prefix)
    snapshot_warc_paths = get_main_warc_paths(
        index, starting_snapshot, ending_snapshot, prefix=data_domain_prefix
    )

    os.makedirs(output_dir, exist_ok=True)

    for snapshot_id, warc_path in snapshot_warc_paths.items():
        urls = fetch_urls_from_warc_path(warc_path, data_domain_prefix)
        output_path = os.path.join(output_dir, f"{snapshot_id}.txt")
        with open(output_path, "w") as f:
            for url in urls:
                f.write(f"{url}\n")
        logger.info(f"Wrote {len(urls)} URLs to {output_path}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--start_snapshot", type=str, default="2025-18")
    parser.add_argument("--end_snapshot", type=str, default="2025-18")
    parser.add_argument("--output_dir", type=str, default="data/cc/url")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    get_common_crawl_urls_per_snapshot(
        args.start_snapshot, args.end_snapshot, output_dir=args.output_dir
    )
