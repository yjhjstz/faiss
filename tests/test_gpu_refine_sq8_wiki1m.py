#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GpuIndexRefine SQ8 测试 - Wiki-1M 数据集
比较 Float16 和 SQ8 的性能、Recall 和显存占用
"""

import numpy as np
import time
import os
import sys
import subprocess

build_lib = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build", "faiss", "python", "build", "lib"
)
if os.path.exists(build_lib):
    for d in os.listdir(build_lib):
        sys.path.insert(0, os.path.join(build_lib, d))

import faiss


def get_gpu_memory_mb():
    """获取当前 GPU 显存使用量 (MB)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip().split('\n')[0])
    except:
        return 0


def load_fbin(filename):
    """加载 .fbin 格式文件"""
    with open(filename, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(nrows, ncols)
    return data


def load_ibin(filename):
    """加载 .ibin 格式文件"""
    with open(filename, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.int32).reshape(nrows, ncols)
    return data


def load_wiki1m(data_dir="wiki_all_1M"):
    """加载 Wiki-1M 数据集"""
    fbin_base = f"{data_dir}/base.1M.fbin"
    fbin_query = f"{data_dir}/queries.fbin"
    ibin_gt = f"{data_dir}/groundtruth.1M.neighbors.ibin"

    if os.path.exists(fbin_base):
        print(f"加载 .fbin 格式数据...")
        xb = load_fbin(fbin_base).astype(np.float32)
        xq = load_fbin(fbin_query).astype(np.float32)
        gt = load_ibin(ibin_gt).astype(np.int64)
        return xb, xq, gt

    raise FileNotFoundError(f"未找到数据文件: {data_dir}")


def compute_recall(I_result, I_gt, k):
    """计算 recall@k"""
    nq = I_result.shape[0]
    recall = 0.0
    for i in range(nq):
        gt_set = set(I_gt[i, :k])
        result_set = set(I_result[i, :k])
        recall += len(gt_set & result_set) / k
    return recall / nq


def test_sq8_vs_float16(xb, xq, gt, config):
    """
    比较 SQ8 和 Float16 的性能
    """
    nb, d = xb.shape
    nq = xq.shape[0]

    nlist = config.get("nlist", 1024)
    m = config.get("m", 48)
    nbits = config.get("nbits", 8)
    nprobe = config.get("nprobe", 32)
    k = config.get("k", 10)
    k_factor = config.get("k_factor", 10)

    print(f"\n{'='*70}")
    print(f"配置: nlist={nlist}, m={m}, nprobe={nprobe}, k_factor={k_factor}, k={k}")
    print(f"数据: nb={nb}, nq={nq}, d={d}")
    print(f"{'='*70}")

    # 理论显存
    float16_mem_theory = nb * d * 2 / 1024 / 1024
    sq8_mem_theory = nb * d * 1 / 1024 / 1024
    print(f"\nRefine 索引理论显存:")
    print(f"  Float16: {float16_mem_theory:.0f} MB")
    print(f"  SQ8:     {sq8_mem_theory:.0f} MB")
    print(f"  节省:    {float16_mem_theory - sq8_mem_theory:.0f} MB ({(1-sq8_mem_theory/float16_mem_theory)*100:.0f}%)")

    # GPU 资源
    res = faiss.StandardGpuResources()
    res.setTempMemory(256 * 1024 * 1024)

    mem_init = get_gpu_memory_mb()
    print(f"\n初始显存: {mem_init} MB")

    # 直接在 GPU 上训练 IVF-PQ 索引
    print("\n创建 GpuIndexIVFPQ 并在 GPU 上训练...")
    train_size = min(nb, nlist * 40)
    t_train = time.time()
    cfg_pq = faiss.GpuIndexIVFPQConfig()
    cfg_pq.useFloat16LookupTables = True
    cfg_pq.device = 0
    index_gpu = faiss.GpuIndexIVFPQ(
        res, d, nlist, m, nbits, faiss.METRIC_L2, cfg_pq)
    index_gpu.train(xb[:train_size])
    index_gpu.add(xb)
    index_gpu.nprobe = nprobe
    print(f"GPU 训练+add 完成，耗时 {time.time() - t_train:.1f}s")

    mem_after_base = get_gpu_memory_mb()
    print(f"Base index 后显存: {mem_after_base} MB (+{mem_after_base - mem_init} MB)")

    # ============ 1. Float16 GpuIndexRefine ============
    print(f"\n--- Float16 GpuIndexRefine ---")

    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = True
    gpu_flat = faiss.GpuIndexFlatL2(res, d, flat_config)
    gpu_flat.add(xb)

    mem_after_f16 = get_gpu_memory_mb()
    print(f"Float16 Refine 后显存: {mem_after_f16} MB (+{mem_after_f16 - mem_after_base} MB)")

    gpu_refine_f16 = faiss.GpuIndexRefine(res, index_gpu, gpu_flat)
    gpu_refine_f16.setKFactor(k_factor)

    # Warmup
    gpu_refine_f16.search(xq[:10], k)

    t0 = time.time()
    D_f16, I_f16 = gpu_refine_f16.search(xq, k)
    time_f16 = time.time() - t0
    recall_f16 = compute_recall(I_f16, gt, k)

    print(f"  搜索时间: {time_f16*1000:.2f} ms")
    print(f"  Recall@{k}: {recall_f16:.4f}")
    print(f"  QPS: {nq/time_f16:.0f}")

    # 释放 Float16
    del gpu_flat
    del gpu_refine_f16
    import gc
    gc.collect()

    mem_after_f16_del = get_gpu_memory_mb()
    print(f"释放 Float16 后显存: {mem_after_f16_del} MB")

    # ============ 2. SQ8 GpuIndexRefine ============
    print(f"\n--- SQ8 GpuIndexRefine ---")

    # SQ8 需要创建新的 base index，通过 GpuIndexRefine.add() 同时添加
    print("  创建新的 GpuIndexIVFPQ base (用于 SQ8)，GPU 上训练...")
    t_train2 = time.time()
    cfg_pq2 = faiss.GpuIndexIVFPQConfig()
    cfg_pq2.useFloat16LookupTables = True
    cfg_pq2.device = 0
    index_gpu2 = faiss.GpuIndexIVFPQ(
        res, d, nlist, m, nbits, faiss.METRIC_L2, cfg_pq2)
    index_gpu2.train(xb[:train_size])
    # 不要在这里 add，让 GpuIndexRefine.add() 来同时添加到 base 和 SQ8
    index_gpu2.nprobe = nprobe
    print(f"  GPU 训练完成，耗时 {time.time() - t_train2:.1f}s")

    cfg = faiss.GpuIndexRefineConfig()
    cfg.storageType = faiss.RefineStorageType_SQ8
    cfg.k_factor = k_factor

    gpu_refine_sq8 = faiss.GpuIndexRefine(res, index_gpu2, cfg)

    # Train SQ8 and add data to both base and SQ8
    print("  训练 SQ8...")
    gpu_refine_sq8.train(xb)
    print("  添加数据 (同时到 base 和 SQ8)...")
    gpu_refine_sq8.add(xb)

    mem_after_sq8 = get_gpu_memory_mb()
    print(f"SQ8 Refine 后显存: {mem_after_sq8} MB (+{mem_after_sq8 - mem_after_f16_del} MB)")

    # Warmup
    gpu_refine_sq8.search(xq[:10], k)

    t0 = time.time()
    D_sq8, I_sq8 = gpu_refine_sq8.search(xq, k)
    time_sq8 = time.time() - t0
    recall_sq8 = compute_recall(I_sq8, gt, k)

    print(f"  搜索时间: {time_sq8*1000:.2f} ms")
    print(f"  Recall@{k}: {recall_sq8:.4f}")
    print(f"  QPS: {nq/time_sq8:.0f}")

    # ============ 结果汇总 ============
    print(f"\n{'='*70}")
    print("结果汇总")
    print(f"{'='*70}")
    print(f"{'方法':<25} {'Recall@'+str(k):<12} {'时间(ms)':<12} {'QPS':<12} {'显存增量':<15}")
    print(f"{'-'*70}")
    print(f"{'Float16':<25} {recall_f16:<12.4f} {time_f16*1000:<12.2f} {nq/time_f16:<12.0f} {mem_after_f16 - mem_after_base:<15} MB")
    print(f"{'SQ8':<25} {recall_sq8:<12.4f} {time_sq8*1000:<12.2f} {nq/time_sq8:<12.0f} {mem_after_sq8 - mem_after_f16_del:<15} MB")
    print(f"{'='*70}")

    print("\n性能对比:")
    print(f"  Recall 差异: {(recall_sq8 - recall_f16)*100:+.2f}%")
    print(f"  时间对比: SQ8 {time_sq8/time_f16:.2f}x (相对 Float16)")
    print(f"  显存节省: ~{float16_mem_theory - sq8_mem_theory:.0f} MB (理论值)")

    return {
        "config": config,
        "recall_f16": recall_f16,
        "recall_sq8": recall_sq8,
        "time_f16": time_f16,
        "time_sq8": time_sq8,
        "mem_f16": mem_after_f16 - mem_after_base,
        "mem_sq8": mem_after_sq8 - mem_after_f16_del,
    }


def main():
    print("=" * 70)
    print("GpuIndexRefine SQ8 测试 - Wiki-1M 数据集")
    print("=" * 70)

    # 查找数据集
    wiki_dirs = ["wiki_all_1M", "tests/wiki_all_1M", "../wiki_all_1M"]

    xb, xq, gt = None, None, None
    for wiki_dir in wiki_dirs:
        if os.path.exists(f"{wiki_dir}/base.1M.fbin"):
            print(f"\n发现 Wiki-1M 数据集: {wiki_dir}")
            xb, xq, gt = load_wiki1m(wiki_dir)
            print(f"数据库: {xb.shape}, 查询: {xq.shape}, GT: {gt.shape}")
            break

    if xb is None:
        print("\n错误: 未找到 Wiki-1M 数据集!")
        sys.exit(1)

    # 使用前 1000 个查询
    nq_test = 1000
    xq = xq[:nq_test]
    gt = gt[:nq_test]

    print(f"\n使用 {nq_test} 个查询进行测试")

    # 测试配置
    config = {"nlist": 1024, "m": 48, "nprobe": 32, "k_factor": 10, "k": 10}

    result = test_sq8_vs_float16(xb, xq, gt, config)

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
