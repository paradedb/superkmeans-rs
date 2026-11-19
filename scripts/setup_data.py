#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pyarrow",
#   "datasets",
#   "huggingface_hub",
#   "faiss-cpu==1.16.0",
# ]
# ///
"""Download Cohere Embed-V3 (10M) and compute exact L2 ground truth.

Usage: uv run --script scripts/setup_data.py [--data-dir data]

Writes, inside --data-dir:
  data_cohere.bin       10M x 1024 vectors: raw little-endian float32, row-major, no header
  data_cohere_test.bin  1,000 queries: same format
  gt_cohere.json        {"query index": [ids of its 100 nearest neighbors]}
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

N_BASE = 10_000_000
D = 1024
N_QUERIES = 1000
KNN = 100
REPO_ID = "Cohere/msmarco-v2.1-embed-english-v3"

ROOT = Path(__file__).resolve().parents[1]


def has_size(path: Path, n_floats: int) -> bool:
    return path.exists() and path.stat().st_size == n_floats * 4


def download_base(data_dir: Path, cache_dir: Path) -> None:
    """First 10M MS MARCO v2.1 passages embedded with Cohere Embed v3."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = data_dir / "data_cohere.bin"
    written = 0
    with open(out, "wb") as f_out:
        for shard in range(32):
            remote = f"passages_parquet/msmarco_v2.1_doc_segmented_{shard:02d}.parquet"
            print(f"Downloading {REPO_ID}/{remote} ...")
            local = hf_hub_download(
                repo_id=REPO_ID,
                filename=remote,
                repo_type="dataset",
                revision="e78737fe92ac1b783211b705c12207ca75fcc9b7",
                cache_dir=cache_dir,
            )
            for batch in pq.ParquetFile(local).iter_batches(batch_size=100_000, columns=["emb"]):
                values = batch.column("emb").flatten().to_numpy()
                vectors = values.astype(np.float32, copy=False).reshape(-1, D)
                take = min(len(vectors), N_BASE - written)
                vectors[:take].tofile(f_out)
                written += take
                if written == N_BASE:
                    break
            print(f"  {written:,} / {N_BASE:,} passages")
            shutil.rmtree(cache_dir, ignore_errors=True)
            if written == N_BASE:
                break

    if written != N_BASE:
        out.unlink(missing_ok=True)
        raise SystemExit(f"Only downloaded {written:,} / {N_BASE:,} vectors")

    print(f"  data_cohere.bin: {written:,} x {D}")


def download_queries(data_dir: Path, cache_dir: Path) -> None:
    """First 1,000 queries of the Cohere MS MARCO test split."""
    from datasets import load_dataset

    print(f"Downloading {REPO_ID} queries ...")
    queries = load_dataset(
        REPO_ID,
        "queries",
        split="test",
        revision="e78737fe92ac1b783211b705c12207ca75fcc9b7",
        cache_dir=str(cache_dir),
    )
    embeddings = np.asarray(queries[:N_QUERIES]["emb"], dtype=np.float32)
    embeddings.tofile(data_dir / "data_cohere_test.bin")
    print(f"  data_cohere_test.bin: {len(embeddings):,} x {embeddings.shape[1]}")
    shutil.rmtree(cache_dir, ignore_errors=True)


def compute_ground_truth(data_dir: Path) -> None:
    """Exact (brute-force L2) top-100 neighbors of the first 1,000 queries."""
    import faiss

    print("Computing ground truth for cohere ...")
    base = np.fromfile(data_dir / "data_cohere.bin", dtype=np.float32).reshape(N_BASE, D)
    queries = np.fromfile(data_dir / "data_cohere_test.bin", dtype=np.float32).reshape(-1, D)
    queries = queries[:N_QUERIES]
    _, ids = faiss.knn(queries, base, KNN)
    with open(data_dir / "gt_cohere.json", "w") as f:
        json.dump({str(i): ids[i].tolist() for i in range(len(queries))}, f)
    print(f"  gt_cohere.json: {len(queries):,} queries x top-{KNN}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Cohere-10M and compute its ground truth"
    )
    parser.add_argument(
        "--data-dir",
        default=ROOT / "data",
        type=Path,
        help="Output directory (default: <repo>/data)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / ".hf_cache"

    if has_size(data_dir / "data_cohere.bin", N_BASE * D):
        print("data_cohere.bin already present, skipping download")
    else:
        download_base(data_dir, cache_dir)

    if has_size(data_dir / "data_cohere_test.bin", N_QUERIES * D):
        print("data_cohere_test.bin already present, skipping download")
    else:
        download_queries(data_dir, cache_dir)

    if (data_dir / "gt_cohere.json").exists():
        print("gt_cohere.json already present, skipping ground truth")
    else:
        compute_ground_truth(data_dir)


if __name__ == "__main__":
    main()
