#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU IVF-PQ Rerank 测试 v3
使用 L2 距离和 Wiki-1M 数据集 (768维文本向量)
"""

import numpy as np
import time
import os
import sys

# 添加 faiss 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss


def download_wiki1m(data_dir="wiki_all_1M"):
    """下载 Wiki-1M 数据集"""
    if os.path.exists(data_dir) and (
        os.path.exists(f"{data_dir}/base.1M.fbin") or
        os.path.exists(f"{data_dir}/base.npy")
    ):
        print(f"Wiki-1M 数据集已存在: {data_dir}")
        return True

    print("下载 Wiki-1M 数据集...")
    tar_file = "wiki_all_1M.tar"

    # 下载
    ret = os.system(f"wget -q --show-progress https://data.rapids.ai/raft/datasets/wiki_all_1M/{tar_file}")
    if ret != 0:
        print("下载失败!")
        return False

    # 解压
    ret = os.system(f"tar xf {tar_file}")
    if ret != 0:
        print("解压失败!")
        return False

    print("下载完成!")
    return True


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
    """
    CPU 上对候选结果进行 rerank (L2 距离)

    Args:
        xq: 查询向量 (nq, d)
        xb: 数据库向量 (nb, d)
        I_candidates: 候选索引 (nq, n_candidates)
        k: 最终返回的结果数

    Returns:
        I_reranked: rerank 后的索引 (nq, k)
        D_reranked: rerank 后的距离 (nq, k)
    """
    nq = xq.shape[0]
    n_candidates = I_candidates.shape[1]

    I_reranked = np.zeros((nq, k), dtype=np.int64)
    D_reranked = np.zeros((nq, k), dtype=np.float32)

    for i in range(nq):
        # 获取候选向量
        candidates_idx = I_candidates[i]
        # 过滤无效索引 (-1)
        valid_mask = candidates_idx >= 0
        valid_idx = candidates_idx[valid_mask]

        if len(valid_idx) == 0:
            I_reranked[i] = -1
            D_reranked[i] = float('inf')
            continue

        candidates = xb[valid_idx]

        # 计算 L2 距离平方 (值越小越相似)
        diff = candidates - xq[i]
        dists = np.sum(diff * diff, axis=1)

        # 取最小的 k 个 (升序)
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


def test_gpu_ivfpq_rerank_l2(xb, xq, gt, config):
    """
    测试 GPU IVF-PQ + CPU Rerank (L2 距离)

    Args:
        xb: 数据库向量 (nb, d)
        xq: 查询向量 (nq, d)
        gt: ground truth (nq, k)
        config: 配置字典 {nlist, m, nprobe, n_candidates}
    """
    nb, d = xb.shape
    nq = xq.shape[0]

    nlist = config.get("nlist", 1024)
    m = config.get("m", 48)
    nbits = config.get("nbits", 8)
    nprobe = config.get("nprobe", 32)
    n_candidates = config.get("n_candidates", 256)
    k = config.get("k", 10)

    print(f"\n{'='*60}")
    print(f"配置: nlist={nlist}, m={m}, nprobe={nprobe}, n_candidates={n_candidates}, k={k}")
    print(f"数据: nb={nb}, nq={nq}, d={d}")
    print(f"{'='*60}")

    # 检查 GPU 可用性
    ngpu = faiss.get_num_gpus()
    if ngpu == 0:
        print("警告: 没有可用的 GPU, 使用 CPU 模式")
        use_gpu = False
    else:
        print(f"检测到 {ngpu} 个 GPU")
        use_gpu = True

    # 创建 CPU 索引 (使用 L2 距离，默认)
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

    if use_gpu:
        # 转换到 GPU
        print("\n转换到 GPU...")
        res = faiss.StandardGpuResources()
        # 使用 float16 查找表减少共享内存使用
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.useFloat16LookupTables = True
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)
        index_gpu.nprobe = nprobe
        index = index_gpu
    else:
        index_cpu.nprobe = nprobe
        index = index_cpu

    # ============ 1. GPU IVF-PQ 无 rerank ============
    print(f"\n1. {'GPU' if use_gpu else 'CPU'} IVF-PQ (无 rerank) - L2")
    t0 = time.time()
    D1, I1 = index.search(xq, k)
    search_time = time.time() - t0
    recall1 = compute_recall(I1, gt, k)
    print(f"   搜索时间: {search_time*1000:.2f}ms")
    print(f"   Recall@{k}: {recall1:.4f}")
    print(f"   QPS: {nq/search_time:.0f}")

    # ============ 2. GPU IVF-PQ + CPU Rerank ============
    print(f"\n2. {'GPU' if use_gpu else 'CPU'} IVF-PQ + CPU Rerank - L2")

    # 获取更多候选
    t0 = time.time()
    D_cand, I_cand = index.search(xq, n_candidates)
    gpu_time = time.time() - t0

    # CPU rerank
    t0 = time.time()
    I2, D2 = cpu_rerank_l2(xq, xb, I_cand, k)
    rerank_time = time.time() - t0

    total_time = gpu_time + rerank_time
    recall2 = compute_recall(I2, gt, k)
    print(f"   GPU 搜索时间: {gpu_time*1000:.2f}ms")
    print(f"   CPU Rerank 时间: {rerank_time*1000:.2f}ms")
    print(f"   总时间: {total_time*1000:.2f}ms")
    print(f"   Recall@{k}: {recall2:.4f}")
    print(f"   QPS: {nq/total_time:.0f}")
    print(f"   Recall 提升: +{(recall2-recall1)*100:.2f}%")

    # ============ 3. IndexRefine 基线 ============
    print(f"\n3. CPU IndexRefine 基线 - L2")

    # 设置 nprobe 在 base_index 上 (关键修复!)
    index_cpu.nprobe = nprobe

    # 创建 flat refine 索引 (使用 L2 度量)
    flat_refine = faiss.IndexFlatL2(d)
    flat_refine.add(xb)

    # 用 IndexRefine 包装 (不是 IndexRefineFlat)
    index_refine = faiss.IndexRefine(index_cpu, flat_refine)
    index_refine.k_factor = n_candidates / k

    t0 = time.time()
    D3, I3 = index_refine.search(xq, k)
    refine_time = time.time() - t0
    recall3 = compute_recall(I3, gt, k)
    print(f"   搜索时间: {refine_time*1000:.2f}ms")
    print(f"   Recall@{k}: {recall3:.4f}")
    print(f"   QPS: {nq/refine_time:.0f}")

    # ============ 4. 精确搜索基线 ============
    print(f"\n4. CPU 精确搜索基线 (IndexFlatL2)")
    index_flat = faiss.IndexFlatL2(d)
    index_flat.add(xb)
    t0 = time.time()
    D4, I4 = index_flat.search(xq, k)
    exact_time = time.time() - t0
    recall4 = compute_recall(I4, gt, k)
    print(f"   搜索时间: {exact_time*1000:.2f}ms")
    print(f"   Recall@{k}: {recall4:.4f}")
    print(f"   QPS: {nq/exact_time:.0f}")

    # 结果汇总
    print(f"\n{'='*60}")
    print("结果汇总:")
    print(f"{'方法':<30} {'Recall@'+str(k):<12} {'时间(ms)':<12} {'QPS':<10}")
    print(f"{'-'*60}")
    print(f"{'IVF-PQ (无 rerank)':<30} {recall1:<12.4f} {search_time*1000:<12.2f} {nq/search_time:<10.0f}")
    print(f"{'IVF-PQ + CPU Rerank':<30} {recall2:<12.4f} {total_time*1000:<12.2f} {nq/total_time:<10.0f}")
    print(f"{'IndexRefine':<30} {recall3:<12.4f} {refine_time*1000:<12.2f} {nq/refine_time:<10.0f}")
    print(f"{'精确搜索':<30} {recall4:<12.4f} {exact_time*1000:<12.2f} {nq/exact_time:<10.0f}")
    print(f"{'='*60}")

    return {
        "config": config,
        "recall_no_rerank": recall1,
        "recall_rerank": recall2,
        "recall_refine": recall3,
        "recall_exact": recall4,
        "time_no_rerank": search_time,
        "time_rerank": total_time,
        "time_refine": refine_time,
        "time_exact": exact_time,
    }


def main():
    print("=" * 60)
    print("GPU IVF-PQ Rerank 测试 v3 - L2 距离")
    print("=" * 60)

    # 尝试加载 Wiki-1M 数据集
    wiki_dir = "wiki_all_1M"
    use_wiki = False

    # 检查 .fbin 或 .npy 格式
    wiki_exists = (
        os.path.exists(f"{wiki_dir}/base.1M.fbin") or
        os.path.exists(f"{wiki_dir}/base.npy")
    )
    if os.path.exists(wiki_dir) and wiki_exists:
        print(f"\n发现 Wiki-1M 数据集: {wiki_dir}")
        try:
            xb, xq, gt = load_wiki1m(wiki_dir)
            print(f"数据库: {xb.shape}, 查询: {xq.shape}, GT: {gt.shape}")
            use_wiki = True
        except Exception as e:
            print(f"加载 Wiki-1M 失败: {e}")
            exit(1)

    if not use_wiki:
        print("\n错误: 未找到 Wiki-1M 数据集!")
        print("请运行以下命令下载数据集:")
        print("  wget https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar")
        print("  tar xf wiki_all_1M.tar")
        sys.exit(1)

    # 只使用前 100 个查询 (加快测试速度)
    nq_test = 100
    xq = xq[:nq_test]
    gt = gt[:nq_test]

    print(f"\n数据库: {xb.shape}, 查询: {xq.shape}")
    print(f"Metric: L2 (使用 Wiki-1M 自带 ground truth)")

    # Wiki-1M 参数 (d=768, m 需要整除 d)
    # 768 = 2^8 * 3, 可被 1,2,3,4,6,8,12,16,24,32,48,64,96... 整除
    # GPU 共享内存限制: m * 256 * 4 <= 49152, 即 m <= 48
    configs = [
        {"nlist": 1024, "m": 32, "nprobe": 32, "n_candidates": 128, "k": 10},
        {"nlist": 1024, "m": 48, "nprobe": 32, "n_candidates": 128, "k": 10},
        {"nlist": 1024, "m": 48, "nprobe": 64, "n_candidates": 256, "k": 10},
        {"nlist": 2048, "m": 48, "nprobe": 64, "n_candidates": 512, "k": 10},
    ]

    results = []
    for config in configs:
        result = test_gpu_ivfpq_rerank_l2(xb, xq, gt, config)
        results.append(result)

    # 最终汇总
    print("\n" + "=" * 80)
    print("所有配置结果汇总")
    print("=" * 80)
    print(f"{'配置':<35} {'无Rerank':<12} {'Rerank':<12} {'提升':<10}")
    print("-" * 80)
    for r in results:
        cfg = r["config"]
        cfg_str = f"nlist={cfg['nlist']},m={cfg['m']},nprobe={cfg['nprobe']}"
        improvement = (r["recall_rerank"] - r["recall_no_rerank"]) * 100
        print(f"{cfg_str:<35} {r['recall_no_rerank']:<12.4f} {r['recall_rerank']:<12.4f} {improvement:+.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
