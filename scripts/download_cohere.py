#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "datasets",
# ]
# ///
"""Download the first N Cohere Embed-V3 (1024-d) vectors as a raw f32 binary.

Source matches the C++ SuperKMeans cohere bench:
  HuggingFace Cohere/msmarco-v2.1-embed-english-v3 (passages)

Default is a 1M prefix (~4.1 GiB) suitable for local Criterion runs.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMBEDDING_DIM = 1024
HF_DATASET = "Cohere/msmarco-v2.1-embed-english-v3"
SHARDS = [
    f"passages_parquet/msmarco_v2.1_doc_segmented_{i:02d}.parquet" for i in range(0, 32)
]


def download(n: int, out: Path) -> None:
    from datasets import load_dataset

    cache_dir = DATA_DIR / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        expected = n * EMBEDDING_DIM * 4
        actual = out.stat().st_size
        if actual == expected:
            print(f"Already present: {out} ({actual:,} bytes)")
            return
        print(f"Removing incomplete/mismatched file ({actual:,} bytes, want {expected:,})")
        out.unlink()

    write_chunk_size = 100_000
    write_buffer = np.empty((write_chunk_size, EMBEDDING_DIM), dtype=np.float32)
    buffer_idx = 0
    total_count = 0

    print(f"Downloading first {n:,} vectors (d={EMBEDDING_DIM}) -> {out}")
    with open(out, "wb") as f_train:
        for shard_path in SHARDS:
            print(f"\nProcessing shard: {shard_path}")
            try:
                ds = load_dataset(
                    HF_DATASET,
                    "passages",
                    split="train",
                    data_files={"train": [shard_path]},
                    streaming=False,
                )
            except Exception as e:
                print(f"Failed shard {shard_path}: {e}")
                continue

            for batch_start in range(0, len(ds), 10_000):
                batch = ds.select(range(batch_start, min(batch_start + 10_000, len(ds))))
                batch_embeddings = np.array(batch["emb"], dtype=np.float32)

                for vec in batch_embeddings:
                    if total_count >= n:
                        break
                    write_buffer[buffer_idx] = vec
                    buffer_idx += 1
                    total_count += 1
                    if buffer_idx == write_chunk_size:
                        write_buffer.tofile(f_train)
                        f_train.flush()
                        buffer_idx = 0
                        print(f"  saved {total_count:,} embeddings...")

                if total_count >= n:
                    break

            # Drop per-shard cache to keep disk usage bounded.
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
                cache_dir.mkdir(parents=True, exist_ok=True)

            if total_count >= n:
                break

        if buffer_idx > 0:
            write_buffer[:buffer_idx].tofile(f_train)
            f_train.flush()

    if total_count < n:
        raise SystemExit(f"Only downloaded {total_count:,} / {n:,} vectors")

    print(f"Done: {total_count:,} x {EMBEDDING_DIM} -> {out} ({out.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        type=int,
        default=1_000_000,
        help="Number of training vectors to download (default: 1000000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DATA_DIR / "data_cohere_1m.bin",
        help="Output raw f32 binary path",
    )
    args = parser.parse_args()
    download(args.n, args.output)


if __name__ == "__main__":
    main()
