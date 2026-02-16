#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GpuIndexRefine 测试 - Wiki-1M 数据集
使用 L2 距离，比较 GPU IVF-PQ、CPU Rerank 和 GpuIndexRefine 的性能
"""

import numpy as np
import time
import os
import sys

# 添加 faiss 路径
build_lib = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build", "faiss", "python", "build", "lib"
)
if os.path.exists(build_lib):
    for d in os.listdir(build_lib):
        sys.path.insert(0, os.path.join(build_lib, d))

import faiss


def load_fbin(filename):
    """加载 .fbin 格式文件 (RAFT/RAPIDS 格式)"""
    with open(filename, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(nrows, ncols)
    return data


def load_ibin(filename):
    """加载 .ibin 格式文件 (ground truth, int32)"""
    with open(filename, 'rb') as f:
        nrows = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        ncols = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        data = np.frombuffer(f.read(), dtype=np.int32).reshape(nrows, ncols)
    return data


def load_wiki1m(data_dir="wiki_all_1M"):
    """加载 Wiki-1M 数据集 (支持 .fbin/.ibin 和 .npy 格式)"""
    # 尝试 .fbin 格式 (RAFT/RAPIDS)
    fbin_base = f"{data_dir}/base.1M.fbin"
    fbin_query = f"{data_dir}/queries.fbin"
    ibin_gt = f"{data_dir}/groundtruth.1M.neighbors.ibin"

    if os.path.exists(fbin_base):
        print(f"加载 .fbin 格式数据...")
        xb = load_fbin(fbin_base).astype(np.float32)
        xq = load_fbin(fbin_query).astype(np.float32)
        gt = load_ibin(ibin_gt).astype(np.int64)
        return xb, xq, gt

    # 尝试 .npy 格式
    npy_base = f"{data_dir}/base.npy"
    if os.path.exists(npy_base):
        print(f"加载 .npy 格式数据...")
        xb = np.load(f"{data_dir}/base.npy").astype(np.float32)
        xq = np.load(f"{data_dir}/query.npy").astype(np.float32)
        gt = np.load(f"{data_dir}/groundtruth.npy").astype(np.int64)
        return xb, xq, gt

    raise FileNotFoundError(f"未找到数据文件: {data_dir}")


def cpu_rerank_l2(xq, xb, I_candidates, k):
    """CPU 上对候选结果进行 rerank (L2 距离)"""
    nq = xq.shape[0]
    n_candidates = I_candidates.shape[1]

    I_reranked = np.zeros((nq, k), dtype=np.int64)
    D_reranked = np.zeros((nq, k), dtype=np.float32)

    for i in range(nq):
        candidates_idx = I_candidates[i]
        valid_mask = candidates_idx >= 0
        valid_idx = candidates_idx[valid_mask]

        if len(valid_idx) == 0:
            I_reranked[i] = -1
            D_reranked[i] = float('inf')
            continue

        candidates = xb[valid_idx]
        diff = candidates - xq[i]
        dists = np.sum(diff * diff, axis=1)

        top_k = min(k, len(dists))
        top_indices = np.argsort(dists)[:top_k]

        I_reranked[i, :top_k] = valid_idx[top_indices]
        D_reranked[i, :top_k] = dists[top_indices]

        if top_k < k:
            I_reranked[i, top_k:] = -1
            D_reranked[i, top_k:] = float('inf')

    return I_reranked, D_reranked


def compute_recall(I_result, I_gt, k):
    """计算 recall@k"""
    nq = I_result.shape[0]
    recall = 0.0
    for i in range(nq):
        gt_set = set(I_gt[i, :k])
        result_set = set(I_result[i, :k])
        recall += len(gt_set & result_set) / k
    return recall / nq


def test_gpu_index_refine(xb, xq, gt, config):
    """
    测试 GpuIndexRefine vs GPU IVF-PQ + CPU Rerank

    Args:
        xb: 数据库向量 (nb, d)
        xq: 查询向量 (nq, d)
        gt: ground truth (nq, k)
        config: 配置字典
    """
    nb, d = xb.shape
    nq = xq.shape[0]

    nlist = config.get("nlist", 1024)
    m = config.get("m", 48)
    nbits = config.get("nbits", 8)
    nprobe = config.get("nprobe", 32)
    k = config.get("k", 10)
    k_factor = config.get("k_factor", 10)  # 直接使用 k_factor

    n_candidates = int(k * k_factor)  # 计算实际候选数
    print(f"\n{'='*70}")
    print(f"配置: nlist={nlist}, m={m}, nprobe={nprobe}, k_factor={k_factor}, k={k}")
    print(f"数据: nb={nb}, nq={nq}, d={d}")
    print(f"实际候选数: {n_candidates}")
    print(f"{'='*70}")

    # 检查 GPU 可用性
    ngpu = faiss.get_num_gpus()
    if ngpu == 0:
        print("错误: 没有可用的 GPU!")
        return None
    print(f"检测到 {ngpu} 个 GPU")

    # 创建 GPU 资源
    res = faiss.StandardGpuResources()

    # 创建 CPU 索引
    print("\n创建 IVF-PQ 索引 (METRIC_L2)...")
    quantizer = faiss.IndexFlatL2(d)
    index_cpu = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # 训练
    print("训练索引...")
    train_size = min(nb, nlist * 40)
    t0 = time.time()
    index_cpu.train(xb[:train_size])
    print(f"训练时间: {time.time() - t0:.2f}s")

    # 添加向量
    print("添加向量...")
    t0 = time.time()
    index_cpu.add(xb)
    print(f"添加时间: {time.time() - t0:.2f}s")

    # 转换到 GPU
    print("\n转换 IVF-PQ 到 GPU...")
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)
    index_gpu.nprobe = nprobe

    # 创建 GPU Flat 索引用于 refinement (使用 float16 节省显存)
    print("创建 GPU Flat 索引用于 refinement (float16)...")
    t0 = time.time()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = True  # 显存减半: 3GB -> 1.5GB
    gpu_flat = faiss.GpuIndexFlatL2(res, d, flat_config)
    gpu_flat.add(xb)
    print(f"GPU Flat 添加时间: {time.time() - t0:.2f}s (float16, 显存约 {xb.nbytes/2/1e9:.2f}GB)")

    # ============ 1. GPU IVF-PQ 无 rerank ============
    print(f"\n1. GPU IVF-PQ (无 rerank) - L2")
    # Warmup
    index_gpu.search(xq[:10], k)

    t0 = time.time()
    D1, I1 = index_gpu.search(xq, k)
    search_time = time.time() - t0
    recall1 = compute_recall(I1, gt, k)
    print(f"   搜索时间: {search_time*1000:.2f}ms")
    print(f"   Recall@{k}: {recall1:.4f}")
    print(f"   QPS: {nq/search_time:.0f}")

    # ============ 2. GPU IVF-PQ + CPU Rerank ============
    print(f"\n2. GPU IVF-PQ + CPU Rerank - L2")

    # Warmup
    index_gpu.search(xq[:10], n_candidates)

    t0 = time.time()
    D_cand, I_cand = index_gpu.search(xq, n_candidates)
    gpu_time = time.time() - t0

    t0 = time.time()
    I2, D2 = cpu_rerank_l2(xq, xb, I_cand, k)
    rerank_time = time.time() - t0

    total_time_cpu = gpu_time + rerank_time
    recall2 = compute_recall(I2, gt, k)
    print(f"   GPU 搜索时间: {gpu_time*1000:.2f}ms")
    print(f"   CPU Rerank 时间: {rerank_time*1000:.2f}ms")
    print(f"   总时间: {total_time_cpu*1000:.2f}ms")
    print(f"   Recall@{k}: {recall2:.4f}")
    print(f"   QPS: {nq/total_time_cpu:.0f}")

    # ============ 3. GpuIndexRefine (GPU IVF-PQ + GPU Rerank) ============
    print(f"\n3. GpuIndexRefine (GPU IVF-PQ + GPU Rerank) - L2")

    # 创建 GpuIndexRefine
    gpu_refine = faiss.GpuIndexRefine(res, index_gpu, gpu_flat)
    gpu_refine.setKFactor(k_factor)
    print(f"   k_factor: {gpu_refine.getKFactor()}")

    # Warmup
    gpu_refine.search(xq[:10], k)

    t0 = time.time()
    D3, I3 = gpu_refine.search(xq, k)
    refine_time = time.time() - t0
    recall3 = compute_recall(I3, gt, k)
    print(f"   搜索时间: {refine_time*1000:.2f}ms")
    print(f"   Recall@{k}: {recall3:.4f}")
    print(f"   QPS: {nq/refine_time:.0f}")

    # ============ 4. CPU IndexRefine 基线 ============
    print(f"\n4. CPU IndexRefine 基线 - L2")
    index_cpu.nprobe = nprobe

    flat_refine_cpu = faiss.IndexFlatL2(d)
    flat_refine_cpu.add(xb)

    index_refine_cpu = faiss.IndexRefine(index_cpu, flat_refine_cpu)
    index_refine_cpu.k_factor = k_factor

    # Warmup
    index_refine_cpu.search(xq[:10], k)

    t0 = time.time()
    D4, I4 = index_refine_cpu.search(xq, k)
    cpu_refine_time = time.time() - t0
    recall4 = compute_recall(I4, gt, k)
    print(f"   搜索时间: {cpu_refine_time*1000:.2f}ms")
    print(f"   Recall@{k}: {recall4:.4f}")
    print(f"   QPS: {nq/cpu_refine_time:.0f}")

    # 结果汇总
    print(f"\n{'='*70}")
    print("结果汇总:")
    print(f"{'方法':<35} {'Recall@'+str(k):<12} {'时间(ms)':<12} {'QPS':<12}")
    print(f"{'-'*70}")
    print(f"{'GPU IVF-PQ (无 rerank)':<35} {recall1:<12.4f} {search_time*1000:<12.2f} {nq/search_time:<12.0f}")
    print(f"{'GPU IVF-PQ + CPU Rerank':<35} {recall2:<12.4f} {total_time_cpu*1000:<12.2f} {nq/total_time_cpu:<12.0f}")
    print(f"{'GpuIndexRefine (GPU Rerank)':<35} {recall3:<12.4f} {refine_time*1000:<12.2f} {nq/refine_time:<12.0f}")
    print(f"{'CPU IndexRefine':<35} {recall4:<12.4f} {cpu_refine_time*1000:<12.2f} {nq/cpu_refine_time:<12.0f}")
    print(f"{'='*70}")

    # 性能对比
    print("\n性能对比:")
    print(f"  GpuIndexRefine vs CPU Rerank: {total_time_cpu/refine_time:.2f}x 加速")
    print(f"  GpuIndexRefine vs CPU IndexRefine: {cpu_refine_time/refine_time:.2f}x 加速")
    print(f"  Recall 差异 (GpuIndexRefine vs CPU Rerank): {(recall3-recall2)*100:+.2f}%")

    return {
        "config": config,
        "recall_no_rerank": recall1,
        "recall_cpu_rerank": recall2,
        "recall_gpu_refine": recall3,
        "recall_cpu_refine": recall4,
        "time_no_rerank": search_time,
        "time_cpu_rerank": total_time_cpu,
        "time_gpu_refine": refine_time,
        "time_cpu_refine": cpu_refine_time,
    }


def main():
    print("=" * 70)
    print("GpuIndexRefine 测试 - Wiki-1M 数据集")
    print("=" * 70)

    # 查找 Wiki-1M 数据集
    wiki_dirs = [
        "wiki_all_1M",
        "tests/wiki_all_1M",
        "../wiki_all_1M",
    ]

    xb, xq, gt = None, None, None
    for wiki_dir in wiki_dirs:
        wiki_exists = (
            os.path.exists(f"{wiki_dir}/base.1M.fbin") or
            os.path.exists(f"{wiki_dir}/base.npy")
        )
        if os.path.exists(wiki_dir) and wiki_exists:
            print(f"\n发现 Wiki-1M 数据集: {wiki_dir}")
            try:
                xb, xq, gt = load_wiki1m(wiki_dir)
                print(f"数据库: {xb.shape}, 查询: {xq.shape}, GT: {gt.shape}")
                break
            except Exception as e:
                print(f"加载失败: {e}")

    if xb is None:
        print("\n错误: 未找到 Wiki-1M 数据集!")
        print("请运行以下命令下载数据集:")
        print("  wget https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar")
        print("  tar xf wiki_all_1M.tar")
        sys.exit(1)

    # 只使用前 1000 个查询 (加快测试速度)
    nq_test = 1000
    xq = xq[:nq_test]
    gt = gt[:nq_test]

    print(f"\n使用 {nq_test} 个查询进行测试")
    print(f"数据库: {xb.shape}, 查询: {xq.shape}")

    # 测试配置 (直接使用 k_factor)
    configs = [
        {"nlist": 1024, "m": 48, "nprobe": 32, "k_factor": 10, "k": 10},   # 100 candidates
        {"nlist": 1024, "m": 48, "nprobe": 64, "k_factor": 25, "k": 10},   # 250 candidates
        {"nlist": 2048, "m": 48, "nprobe": 64, "k_factor": 50, "k": 10},   # 500 candidates
    ]

    results = []
    for config in configs:
        result = test_gpu_index_refine(xb, xq, gt, config)
        if result:
            results.append(result)

    # 最终汇总
    if results:
        print("\n" + "=" * 80)
        print("所有配置结果汇总")
        print("=" * 80)
        print(f"{'配置':<40} {'无Rerank':<10} {'CPU Rerank':<12} {'GPU Refine':<12} {'加速比':<10}")
        print("-" * 80)
        for r in results:
            cfg = r["config"]
            cfg_str = f"nlist={cfg['nlist']},m={cfg['m']},np={cfg['nprobe']},kf={cfg['k_factor']}"
            speedup = r["time_cpu_rerank"] / r["time_gpu_refine"]
            print(f"{cfg_str:<40} {r['recall_no_rerank']:<10.4f} {r['recall_cpu_rerank']:<12.4f} {r['recall_gpu_refine']:<12.4f} {speedup:<10.2f}x")
        print("=" * 80)


if __name__ == "__main__":
    main()
