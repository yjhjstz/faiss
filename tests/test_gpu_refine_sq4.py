#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GpuIndexRefine SQ4 测试
比较 Float16 和 SQ4 的性能、Recall 和显存占用
（参考 tests/test_gpu_refine_sq8_wiki1m.py）
"""

import numpy as np
import time
import os
import sys
import subprocess
import gc

build_lib = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build", "faiss", "python", "build", "lib"
)
if os.path.exists(build_lib):
    for d in os.listdir(build_lib):
        sys.path.insert(0, os.path.join(build_lib, d))

import faiss


def get_gpu_memory_mb():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip().split('\n')[0])
    except Exception:
        return 0


def load_fbin(filename):
    with open(filename, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(nrows, ncols)
    return data


def load_ibin(filename):
    with open(filename, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.int32).reshape(nrows, ncols)
    return data


def load_wiki1m(data_dir):
    fbin_base = f"{data_dir}/base.1M.fbin"
    fbin_query = f"{data_dir}/queries.fbin"
    ibin_gt = f"{data_dir}/groundtruth.1M.neighbors.ibin"

    xb = load_fbin(fbin_base).astype(np.float32)
    xq = load_fbin(fbin_query).astype(np.float32)
    gt = load_ibin(ibin_gt).astype(np.int64)
    return xb, xq, gt


def compute_recall(I_result, I_gt, k):
    nq = I_result.shape[0]
    recall = 0.0
    for i in range(nq):
        gt_set = set(I_gt[i, :k])
        result_set = set(I_result[i, :k])
        recall += len(gt_set & result_set) / k
    return recall / nq


def synthetic_dataset(nb=50000, nq=200, d=128, seed=1234):
    """Fallback dataset when wiki-1M isn't available."""
    rs = np.random.RandomState(seed)
    xb = rs.randn(nb, d).astype('float32')
    xq = rs.randn(nq, d).astype('float32')
    # brute-force ground truth
    idx = faiss.IndexFlatL2(d)
    idx.add(xb)
    _, gt = idx.search(xq, 100)
    return xb, xq, gt.astype(np.int64)


def build_refine(res, index_gpu_base, xb, d, storage):
    """Build a GpuIndexRefine with the requested storage type."""
    if storage == "float16":
        flat_cfg = faiss.GpuIndexFlatConfig()
        flat_cfg.useFloat16 = True
        flat = faiss.GpuIndexFlatL2(res, d, flat_cfg)
        flat.add(xb)
        return faiss.GpuIndexRefine(res, index_gpu_base, flat), flat

    cfg = faiss.GpuIndexRefineConfig()
    if storage == "sq8":
        cfg.storageType = faiss.RefineStorageType_SQ8
    elif storage == "sq4":
        cfg.storageType = faiss.RefineStorageType_SQ4
    else:
        raise ValueError(f"unknown storage: {storage}")
    cfg.k_factor = 1.0  # will be overridden later

    refine = faiss.GpuIndexRefine(res, index_gpu_base, cfg)
    refine.train(xb)
    refine.add(xb)
    return refine, None


def bench(xb, xq, gt, config):
    nb, d = xb.shape
    nq = xq.shape[0]

    nlist = config.get("nlist", 1024)
    m = config.get("m", 16)
    nbits = config.get("nbits", 8)
    nprobe = config.get("nprobe", 32)
    k = config.get("k", 10)
    k_factor = config.get("k_factor", 10)

    print(f"\n{'=' * 70}")
    print(f"配置: nlist={nlist}, m={m}, nprobe={nprobe}, "
          f"k_factor={k_factor}, k={k}")
    print(f"数据: nb={nb}, nq={nq}, d={d}")
    print(f"{'=' * 70}")

    # 理论显存
    f16_mb = nb * d * 2 / 1024 / 1024
    sq8_mb = nb * d * 1 / 1024 / 1024
    sq4_mb = nb * ((d + 1) // 2) * 1 / 1024 / 1024
    print(f"\nRefine 索引理论显存:")
    print(f"  Float16: {f16_mb:.0f} MB")
    print(f"  SQ8:     {sq8_mb:.0f} MB ({(1 - sq8_mb / f16_mb) * 100:.0f}% 节省)")
    print(f"  SQ4:     {sq4_mb:.0f} MB ({(1 - sq4_mb / f16_mb) * 100:.0f}% 节省)")

    res = faiss.StandardGpuResources()
    res.setTempMemory(256 * 1024 * 1024)

    results = {}

    def run_variant(name, storage):
        print(f"\n--- {name} ---", flush=True)
        mem_before = get_gpu_memory_mb()

        print(f"  [{name}] 构建 GpuIndexIVFPQ 并在 GPU 上训练 ...",
              flush=True)
        t_train = time.time()
        cfg_pq = faiss.GpuIndexIVFPQConfig()
        cfg_pq.useFloat16LookupTables = True
        cfg_pq.device = 0
        index_gpu = faiss.GpuIndexIVFPQ(
            res, d, nlist, m, nbits, faiss.METRIC_L2, cfg_pq)
        train_size = min(nb, nlist * 40)
        index_gpu.train(xb[:train_size])
        if storage == "float16":
            index_gpu.add(xb)  # float16 路径：base 先加数据
        index_gpu.nprobe = nprobe
        print(f"  [{name}] GPU 训练完成，耗时 {time.time() - t_train:.1f}s",
              flush=True)
        # 占位兼容下面 del 语句
        index_cpu = None

        print(f"  [{name}] 构建 refine ({storage}) ...", flush=True)
        t_refine = time.time()
        refine, _ = build_refine(res, index_gpu, xb, d, storage)
        refine.setKFactor(k_factor)
        print(f"  [{name}] refine 构建完成，耗时 "
              f"{time.time() - t_refine:.1f}s", flush=True)

        mem_after = get_gpu_memory_mb()
        print(f"  [{name}] 显存增量: {mem_after - mem_before} MB", flush=True)

        # warmup
        refine.search(xq[:10], k)

        t0 = time.time()
        D, I = refine.search(xq, k)
        dt = time.time() - t0
        recall = compute_recall(I, gt, k)

        print(f"  [{name}] 搜索时间: {dt * 1000:.2f} ms", flush=True)
        print(f"  [{name}] Recall@{k}: {recall:.4f}", flush=True)
        print(f"  [{name}] QPS: {nq / dt:.0f}", flush=True)

        results[name] = dict(recall=recall, time=dt,
                             mem=mem_after - mem_before)

        del refine, index_gpu, index_cpu
        gc.collect()

    run_variant("Float16", "float16")
    run_variant("SQ8", "sq8")
    run_variant("SQ4", "sq4")

    print(f"\n{'=' * 70}")
    print("结果汇总")
    print(f"{'=' * 70}")
    header = f"{'方法':<10} {'Recall@' + str(k):<12} {'时间(ms)':<12} " \
             f"{'QPS':<10} {'显存增量':<12}"
    print(header)
    print('-' * 70)
    for name in ("Float16", "SQ8", "SQ4"):
        r = results[name]
        print(f"{name:<10} {r['recall']:<12.4f} {r['time'] * 1000:<12.2f} "
              f"{nq / r['time']:<10.0f} {r['mem']:<12} MB")
    print('=' * 70)

    base = results["Float16"]
    for name in ("SQ8", "SQ4"):
        r = results[name]
        print(f"\n{name} vs Float16:")
        print(f"  Recall 差异: {(r['recall'] - base['recall']) * 100:+.2f} 点")
        print(f"  时间比率:    {r['time'] / base['time']:.2f}x")


def main():
    print("=" * 70)
    print("GpuIndexRefine SQ4 测试")
    print("=" * 70)

    wiki_dirs = ["wiki_all_1M", "tests/wiki_all_1M", "../wiki_all_1M"]
    xb, xq, gt = None, None, None
    for wdir in wiki_dirs:
        if os.path.exists(f"{wdir}/base.1M.fbin"):
            print(f"\n发现 Wiki-1M 数据集: {wdir}")
            xb, xq, gt = load_wiki1m(wdir)
            print(f"数据库: {xb.shape}, 查询: {xq.shape}, GT: {gt.shape}")
            break

    if xb is None:
        print("\n未找到 Wiki-1M, 使用合成数据集...")
        xb, xq, gt = synthetic_dataset()

    nq_test = min(1000, xq.shape[0])
    xq = xq[:nq_test]
    gt = gt[:nq_test]
    print(f"\n使用 {nq_test} 个查询进行测试")

    d = xb.shape[1]
    # m must divide d; pick a safe default for either dataset
    for cand in (48, 32, 16, 8, 4):
        if d % cand == 0:
            m_val = cand
            break
    config = {
        "nlist": 1024, "m": m_val, "nprobe": 32,
        "k_factor": 10, "k": 10,
    }
    bench(xb, xq, gt, config)

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
