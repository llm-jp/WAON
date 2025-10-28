import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/cc/images_size-clean_nsfw-clean_phash_deduplicated_clip_filtered_merged_reverse/2024-26",
        help="Directory containing input files.",
    )
    return parser.parse_args()


def count_jsonl_lines_fast(file_path):
    """JSONLファイルの行数を高速カウント（パースなし）"""
    try:
        with open(file_path, "rb") as f:
            count = sum(1 for _ in f)
        return count
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def count_parquet_rows_fast(file_path):
    """Parquetファイルの行数を高速カウント（メタデータのみ）"""
    try:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def count_jsonl_lines_chunk(file_path, chunk_size=8192):
    """チャンク読み込みでJSONL行数をカウント"""
    try:
        count = 0
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                count += chunk.count(b"\n")
        return count
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def process_file_optimized(file_path):
    """ファイル形式に応じて最適化された処理を実行"""
    file_ext = Path(file_path).suffix.lower()

    if file_ext == ".jsonl":
        # JSONLは行数カウントのみ（パースしない）
        return count_jsonl_lines_chunk(file_path)
    elif file_ext == ".parquet":
        # Parquetはメタデータから行数取得
        return count_parquet_rows_fast(file_path)
    else:
        print(f"Unsupported file format: {file_ext}")
        return 0


def get_supported_files(input_dir, extensions=(".jsonl", ".parquet")):
    """サポートされているファイル形式のファイルリストを取得"""
    input_path = Path(input_dir)
    files = []

    for ext in extensions:
        files.extend(input_path.glob(f"*{ext}"))

    return [str(f) for f in files]


def count_records_parallel(input_dir, max_workers=None, batch_size=None):
    """並列処理でレコード数をカウント"""

    # CPU数に基づいてワーカー数を決定
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 16)  # 最大16プロセス

    # ファイルリスト取得
    input_files = get_supported_files(input_dir)

    if not input_files:
        print(f"No supported files found in {input_dir}")
        return 0

    print(f"Found {len(input_files)} files")
    print(f"Using {max_workers} workers")

    total_count = 0

    # バッチ処理でメモリ使用量を制御
    if batch_size is None:
        batch_size = max(1, len(input_files) // max_workers)

    # 並列処理実行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit使用でより細かい制御
        future_to_file = {
            executor.submit(process_file_optimized, file_path): file_path
            for file_path in input_files
        }

        # 進行状況表示付きで結果収集
        with tqdm(total=len(input_files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    count = future.result()
                    total_count += count
                    pbar.set_postfix({"Total": f"{total_count:,}"})
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                finally:
                    pbar.update(1)

    return total_count


def main(input_dir):
    # 高速化されたカウント実行
    total_records = count_records_parallel(
        input_dir=input_dir,
        max_workers=None,  # 自動決定
        batch_size=None,  # 自動決定
    )

    print(f"\nTotal number of records across all files: {total_records:,}")


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir)


# パフォーマンステスト用の関数
def benchmark_methods(file_path):
    """異なるカウント方法のパフォーマンス比較"""
    import time

    if not Path(file_path).suffix.lower() == ".jsonl":
        print("Benchmark is for JSONL files only")
        return

    methods = {
        "pandas": lambda: len(pd.read_json(file_path, lines=True)),
        "line_count": lambda: count_jsonl_lines_fast(file_path),
        "chunk_count": lambda: count_jsonl_lines_chunk(file_path),
    }

    results = {}
    for name, method in methods.items():
        start_time = time.time()
        try:
            count = method()
            elapsed = time.time() - start_time
            results[name] = {"count": count, "time": elapsed}
            print(f"{name}: {count:,} records in {elapsed:.2f}s")
        except Exception as e:
            print(f"{name}: Error - {e}")

    return results
