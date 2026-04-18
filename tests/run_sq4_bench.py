#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run SQ4 benchmark and batch-scaling measurements and merge them into
the existing JSON result files:
  - tests/benchmark_results.json   (appends one SQ4 entry per dataset)
  - tests/batch_scaling_results.json (adds a "sq4" series per dataset)
"""

import gc
import json
import os
import subprocess
import sys
import time

import numpy as np

build_lib = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build", "faiss", "python", "build", "lib"
)
if os.path.exists(build_lib):
    for d in os.listdir(build_lib):
        sys.path.insert(0, os.path.join(build_lib, d))

import faiss  # noqa: E402


def get_gpu_memory_mb():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True)
        return int(r.stdout.strip().split('\n')[0])
    except Exception:
        return 0


def load_fvecs(fn):
    with open(fn, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, d + 1)[:, 1:]
    return data.astype(np.float32)


def load_ivecs(fn):
    with open(fn, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.seek(0)
        data = np.frombuffer(f.read(), dtype=np.int32).reshape(-1, d + 1)[:, 1:]
    return data


def load_fbin(fn):
    with open(fn, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(nrows, ncols)
    return data


def load_ibin(fn):
    with open(fn, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.int32).reshape(nrows, ncols)
    return data


def compute_recall(I_result, I_gt, k):
    nq = I_result.shape[0]
    acc = 0.0
    for i in range(nq):
        acc += len(set(I_gt[i, :k]) & set(I_result[i, :k])) / k
    return acc / nq


def build_sq4_refine(res, xb, nlist, m, nprobe, k_factor):
    """Train IVF-PQ directly on GPU, then attach SQ4 refine storage."""
    nb, d = xb.shape

    cfg_pq = faiss.GpuIndexIVFPQConfig()
    cfg_pq.useFloat16LookupTables = True
    cfg_pq.device = 0
    base = faiss.GpuIndexIVFPQ(
        res, d, nlist, m, 8, faiss.METRIC_L2, cfg_pq)
    train_size = min(nb, nlist * 40)
    base.train(xb[:train_size])
    base.nprobe = nprobe

    cfg = faiss.GpuIndexRefineConfig()
    cfg.storageType = faiss.RefineStorageType_SQ4
    cfg.k_factor = k_factor

    refine = faiss.GpuIndexRefine(res, base, cfg)
    refine.train(xb)
    refine.add(xb)
    return refine, base


def run_for_dataset(name, xb, xq, gt, m):
    print(f"\n{'=' * 70}\n[{name}] SQ4 benchmark\n{'=' * 70}", flush=True)

    res = faiss.StandardGpuResources()
    res.setTempMemory(256 * 1024 * 1024)

    # GPU warmup for stable VRAM reading
    dummy = faiss.GpuIndexFlatL2(res, xb.shape[1])
    dummy.add(np.zeros((100, xb.shape[1]), dtype=np.float32))
    dummy.search(np.zeros((10, xb.shape[1]), dtype=np.float32), 10)
    del dummy
    gc.collect()

    nq = xq.shape[0]
    k, k_factor, nprobe, nlist = 10, 10, 32, 1024

    # Build the refine index once; reuse it for the single bench and the
    # batch-scaling sweep so we don't re-train / re-encode twice.
    print(f"[{name}] building GPU-trained IVF-PQ + SQ4 refine ...",
          flush=True)
    t0 = time.time()
    mem_before = get_gpu_memory_mb()
    refine, base = build_sq4_refine(res, xb, nlist, m, nprobe, k_factor)
    mem_after = get_gpu_memory_mb()
    vram = mem_after - mem_before
    print(f"  built in {time.time() - t0:.1f}s, VRAM +{vram} MB",
          flush=True)

    # Single-config bench (matches benchmark_results.json schema)
    refine.search(xq[:10], k)  # warmup
    times = []
    for _ in range(3):
        t1 = time.time()
        D, I = refine.search(xq, k)
        times.append(time.time() - t1)
    avg = float(np.mean(times))
    recall = compute_recall(I, gt, k)
    single = {
        "config": {"dataset": name, "nlist": nlist, "m": m, "nprobe": nprobe,
                   "k": k, "k_factor": k_factor, "storage": "sq4"},
        "dataset": name,
        "vram_mb": int(vram),
        "latency_ms": avg * 1000,
        "qps": nq / avg,
        "recall": recall,
    }
    print(f"  Recall@10={recall:.4f}  latency={avg*1000:.2f}ms  "
          f"QPS={nq/avg:.0f}  VRAM={vram}MB", flush=True)

    # Batch-scaling sweep (reuses the same refine index)
    print(f"\n[{name}] batch-scaling sweep (sq4) ...", flush=True)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    batch_sizes = [b for b in batch_sizes if b <= xq.shape[0]]
    series = []
    for bs in batch_sizes:
        q = xq[:bs]
        refine.search(q, k)  # per-size warmup
        ts = []
        for _ in range(5):
            t1 = time.time()
            refine.search(q, k)
            ts.append(time.time() - t1)
        avg = float(np.mean(ts))
        std = float(np.std(ts))
        series.append({
            "batch_size": bs,
            "latency_ms": avg * 1000,
            "latency_std_ms": std * 1000,
            "qps": bs / avg,
            "latency_per_query_us": avg * 1_000_000 / bs,
        })
        print(f"    batch={bs:4d}: {avg*1000:8.2f} ms, "
              f"{bs/avg:10.0f} QPS, {avg*1e6/bs:6.1f} us/q", flush=True)

    del refine, base
    gc.collect()

    return single, series


def merge_json(path, update_fn):
    with open(path, 'r') as f:
        data = json.load(f)
    update_fn(data)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  updated {path}", flush=True)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    results = {}  # name -> (single, series)

    # SIFT1M
    sift_dir = "tests/sift"
    if os.path.exists(f"{sift_dir}/sift_base.fvecs"):
        print("Loading SIFT1M ...", flush=True)
        xb = load_fvecs(f"{sift_dir}/sift_base.fvecs")
        xq = load_fvecs(f"{sift_dir}/sift_query.fvecs")
        gt = load_ivecs(f"{sift_dir}/sift_groundtruth.ivecs").astype(np.int64)
        xq = xq[:1000]
        gt = gt[:1000]
        results["SIFT1M"] = run_for_dataset("SIFT1M", xb, xq, gt, m=16)
        del xb, xq, gt
        gc.collect()
    else:
        print("SIFT1M not found, skipping.", flush=True)

    # Wiki1M
    wiki_dir = "tests/wiki_all_1M"
    if os.path.exists(f"{wiki_dir}/base.1M.fbin"):
        print("Loading Wiki1M ...", flush=True)
        xb = load_fbin(f"{wiki_dir}/base.1M.fbin").astype(np.float32)
        xq = load_fbin(f"{wiki_dir}/queries.fbin").astype(np.float32)
        gt = load_ibin(
            f"{wiki_dir}/groundtruth.1M.neighbors.ibin").astype(np.int64)
        if xq.shape[0] < 1000:
            xq = np.tile(xq, (1000 // xq.shape[0] + 1, 1))[:1000]
            gt = np.tile(gt, (1000 // gt.shape[0] + 1, 1))[:1000]
        else:
            xq = xq[:1000]
            gt = gt[:1000]
        results["Wiki1M"] = run_for_dataset("Wiki1M", xb, xq, gt, m=48)
        del xb, xq, gt
        gc.collect()
    else:
        print("Wiki1M not found, skipping.", flush=True)

    # Merge into benchmark_results.json
    bench_path = "tests/benchmark_results.json"

    def _append_single(data):
        for name, (single, _) in results.items():
            lst = data.setdefault(name, [])
            # Drop any previous SQ4 entry with same knobs
            lst[:] = [
                e for e in lst
                if not (e.get("config", {}).get("storage") == "sq4"
                        and e.get("config", {}).get("nprobe") == 32
                        and e.get("config", {}).get("k_factor") == 10)
            ]
            lst.append(single)

    merge_json(bench_path, _append_single)

    # Merge into batch_scaling_results.json
    bs_path = "tests/batch_scaling_results.json"

    def _attach_series(data):
        for name, (_, series) in results.items():
            block = data.setdefault(name, {
                "dataset": name,
                "config": {"nlist": 1024, "m": 48 if name == "Wiki1M" else 16,
                           "nprobe": 32, "k": 10, "k_factor": 10},
            })
            block["sq4"] = series

    merge_json(bs_path, _attach_series)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
