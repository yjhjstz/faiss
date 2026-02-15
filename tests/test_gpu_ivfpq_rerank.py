# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU IVF-PQ + CPU Rerank baseline test.

This test compares:
1. GPU IVF-PQ without rerank
2. GPU IVF-PQ + CPU exact rerank
3. CPU IndexRefine baseline
"""

import numpy as np
import faiss
import time
import os
import struct


# ========== 读取 texmex 格式 ==========
def read_fvecs(filename):
    with open(filename, 'rb') as f:
        data = []
        while True:
            buf = f.read(4)
            if not buf:
                break
            d = struct.unpack('i', buf)[0]
            vec = struct.unpack('f' * d, f.read(4 * d))
            data.append(vec)
    return np.array(data, dtype='float32')


def read_ivecs(filename):
    with open(filename, 'rb') as f:
        data = []
        while True:
            buf = f.read(4)
            if not buf:
                break
            d = struct.unpack('i', buf)[0]
            vec = struct.unpack('i' * d, f.read(4 * d))
            data.append(vec)
    return np.array(data, dtype='int32')


def recall_at_k(I_pred, I_gt, k):
    """计算 Recall@k"""
    n = I_pred.shape[0]
    hits = 0
    for i in range(n):
        hits += len(set(I_pred[i, :k].tolist()) & set(I_gt[i, :k].tolist()))
    return hits / (n * k)


def cpu_rerank_l2(xq, xb, I_candidates, k):
    """
    CPU 精确 L2 距离 rerank

    Args:
        xq: 查询向量 (nq, d)
        xb: 数据库向量 (nb, d)
        I_candidates: 候选 ID (nq, k_candidates)
        k: 最终返回的 top-k

    Returns:
        D_final: 精确距离 (nq, k)
        I_final: 精确排序后的 ID (nq, k)
    """
    nq = xq.shape[0]
    k_candidates = I_candidates.shape[1]

    D_exact = np.zeros((nq, k_candidates), dtype='float32')

    # 批量计算精确 L2 距离
    for i in range(nq):
        # 处理无效 ID (-1)
        valid_mask = I_candidates[i] >= 0
        valid_ids = I_candidates[i][valid_mask]

        if len(valid_ids) > 0:
            candidates = xb[valid_ids]
            dists = np.sum((xq[i] - candidates) ** 2, axis=1)
            D_exact[i, valid_mask] = dists
            D_exact[i, ~valid_mask] = np.inf
        else:
            D_exact[i] = np.inf

    # 排序取 top-k
    I_final = np.zeros((nq, k), dtype='int64')
    D_final = np.zeros((nq, k), dtype='float32')

    for i in range(nq):
        idx = np.argsort(D_exact[i])[:k]
        I_final[i] = I_candidates[i][idx]
        D_final[i] = D_exact[i][idx]

    return D_final, I_final


def cpu_rerank_l2_batch(xq, xb, I_candidates, k):
    """
    CPU 精确 L2 距离 rerank (优化的批量版本)
    """
    nq = xq.shape[0]
    k_candidates = I_candidates.shape[1]

    # 批量获取候选向量并计算距离
    # 使用矩阵运算加速
    D_exact = np.zeros((nq, k_candidates), dtype='float32')

    for i in range(nq):
        valid_mask = I_candidates[i] >= 0
        valid_ids = I_candidates[i][valid_mask]

        if len(valid_ids) > 0:
            candidates = xb[valid_ids]
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
            diff = xq[i:i+1] - candidates
            dists = np.einsum('ij,ij->i', diff, diff)
            D_exact[i, valid_mask] = dists
            D_exact[i, ~valid_mask] = np.inf
        else:
            D_exact[i] = np.inf

    # 使用 argpartition 加速 top-k 选择
    I_final = np.zeros((nq, k), dtype='int64')
    D_final = np.zeros((nq, k), dtype='float32')

    for i in range(nq):
        if k < k_candidates:
            idx = np.argpartition(D_exact[i], k)[:k]
            idx = idx[np.argsort(D_exact[i][idx])]
        else:
            idx = np.argsort(D_exact[i])[:k]
        I_final[i] = I_candidates[i][idx]
        D_final[i] = D_exact[i][idx]

    return D_final, I_final


def main():
    # ========== 下载 SIFT1M ==========
    if not os.path.exists("sift"):
        os.system("wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz")
        os.system("tar xzf sift.tar.gz")

    print("加载数据...")
    xb = read_fvecs("sift/sift_base.fvecs")
    xq = read_fvecs("sift/sift_query.fvecs")
    gt = read_ivecs("sift/sift_groundtruth.ivecs")

    # 只使用前 1000 个查询
    nq_test = 1000
    xq = xq[:nq_test]
    gt = gt[:nq_test]

    d = xb.shape[1]
    nb = xb.shape[0]
    nq = xq.shape[0]
    k = 10

    print(f"数据库: {xb.shape}, 查询: {xq.shape}, d={d}")

    # GPU 资源
    res = faiss.StandardGpuResources()

    # ========== 测试配置 ==========
    # 基础 IVF-PQ 参数
    base_configs = [
        {"nlist": 1024, "m": 8,  "nprobe": 16},
        {"nlist": 1024, "m": 16, "nprobe": 32},
        {"nlist": 1024, "m": 16, "nprobe": 64},
    ]

    # Rerank k_factor 参数
    k_factors = [4, 10]  # 分别对应 40 和 100 候选

    # ========== 1. GPU IVF-PQ 无 rerank ==========
    print("\n" + "=" * 80)
    print("1. GPU IVF-PQ (无 rerank)")
    print("=" * 80)
    print(f"\n{'nlist':>6} {'m':>4} {'nprobe':>8} {'Recall@1':>10} {'Recall@10':>11} {'QPS':>10} {'ms':>8}")
    print("-" * 65)

    for cfg in base_configs:
        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, cfg["nlist"], cfg["m"], 8)
        cpu_index.train(xb)
        cpu_index.add(xb)

        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        faiss.GpuParameterSpace().set_index_parameter(gpu_index, "nprobe", cfg["nprobe"])

        # 预热
        gpu_index.search(xq[:100], k)

        # 搜索
        t0 = time.time()
        D, I = gpu_index.search(xq, k)
        elapsed_ms = (time.time() - t0) * 1000

        r1 = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, 10)
        qps = nq / (elapsed_ms / 1000)

        print(f"{cfg['nlist']:>6} {cfg['m']:>4} {cfg['nprobe']:>8} {r1:>10.4f} {r10:>11.4f} {qps:>10.0f} {elapsed_ms:>8.2f}")

        del gpu_index, cpu_index

    # ========== 2. GPU IVF-PQ + CPU Rerank ==========
    print("\n" + "=" * 80)
    print("2. GPU IVF-PQ + CPU Rerank")
    print("=" * 80)
    print(f"\n{'nlist':>6} {'m':>4} {'nprobe':>8} {'k_factor':>9} {'Recall@1':>10} {'Recall@10':>11} {'QPS':>10} {'total_ms':>10} {'rerank_ms':>10}")
    print("-" * 97)

    for cfg in base_configs:
        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, cfg["nlist"], cfg["m"], 8)
        cpu_index.train(xb)
        cpu_index.add(xb)

        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        faiss.GpuParameterSpace().set_index_parameter(gpu_index, "nprobe", cfg["nprobe"])

        # 预热
        gpu_index.search(xq[:100], k * max(k_factors))

        for k_factor in k_factors:
            k_search = k * k_factor

            # GPU 搜索候选
            t0 = time.time()
            D_approx, I_candidates = gpu_index.search(xq, k_search)
            gpu_ms = (time.time() - t0) * 1000

            if k_factor == 1:
                # 无 rerank
                I_final = I_candidates
                rerank_ms = 0
            else:
                # CPU rerank
                t1 = time.time()
                D_final, I_final = cpu_rerank_l2_batch(xq, xb, I_candidates, k)
                rerank_ms = (time.time() - t1) * 1000

            total_ms = gpu_ms + rerank_ms

            r1 = recall_at_k(I_final, gt, 1)
            r10 = recall_at_k(I_final, gt, 10)
            qps = nq / (total_ms / 1000)

            print(f"{cfg['nlist']:>6} {cfg['m']:>4} {cfg['nprobe']:>8} {k_factor:>9} {r1:>10.4f} {r10:>11.4f} {qps:>10.0f} {total_ms:>10.2f} {rerank_ms:>10.2f}")

        del gpu_index, cpu_index
        print()  # 分隔不同配置

    # ========== 3. CPU IndexRefine 基线 ==========
    print("\n" + "=" * 80)
    print("3. CPU IndexRefine 基线 (IVF-PQ + Flat rerank)")
    print("=" * 80)
    print(f"\n{'nlist':>6} {'m':>4} {'nprobe':>8} {'k_factor':>9} {'Recall@1':>10} {'Recall@10':>11} {'QPS':>10} {'ms':>8}")
    print("-" * 80)

    refine_configs = [
        {"nlist": 1024, "m": 16, "nprobe": 32, "k_factor": 4},
        {"nlist": 1024, "m": 16, "nprobe": 32, "k_factor": 10},
        {"nlist": 1024, "m": 16, "nprobe": 64, "k_factor": 4},
        {"nlist": 1024, "m": 16, "nprobe": 64, "k_factor": 10},
    ]

    for cfg in refine_configs:
        quantizer = faiss.IndexFlatL2(d)
        ivfpq = faiss.IndexIVFPQ(quantizer, d, cfg["nlist"], cfg["m"], 8)
        ivfpq.train(xb)
        ivfpq.add(xb)
        ivfpq.nprobe = cfg["nprobe"]

        # 用 IndexRefine 包装
        flat_refine = faiss.IndexFlatL2(d)
        flat_refine.add(xb)
        refine_index = faiss.IndexRefine(ivfpq, flat_refine)
        refine_index.k_factor = cfg["k_factor"]

        # 预热
        refine_index.search(xq[:10], k)

        # 搜索
        t0 = time.time()
        D, I = refine_index.search(xq, k)
        elapsed_ms = (time.time() - t0) * 1000

        r1 = recall_at_k(I, gt, 1)
        r10 = recall_at_k(I, gt, 10)
        qps = nq / (elapsed_ms / 1000)

        print(f"{cfg['nlist']:>6} {cfg['m']:>4} {cfg['nprobe']:>8} {cfg['k_factor']:>9} {r1:>10.4f} {r10:>11.4f} {qps:>10.0f} {elapsed_ms:>8.1f}")

        del refine_index, ivfpq

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
