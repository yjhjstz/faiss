/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/RerankKernels.cuh>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <cublas_v2.h>

namespace faiss {
namespace gpu {

// Kernel to compute L2 norms for candidates grouped by query
// Each thread handles one candidate vector
template <int ThreadsPerBlock>
__global__ void computeCandidateNormsKernel(
        const float* __restrict__ candidates, // (n * k_base, d)
        float* __restrict__ norms,            // (n * k_base)
        idx_t totalCandidates,
        int d) {
    idx_t idx = blockIdx.x * ThreadsPerBlock + threadIdx.x;
    if (idx >= totalCandidates) return;

    const float* vec = candidates + idx * d;
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        float v = vec[i];
        sum += v * v;
    }
    norms[idx] = sum;
}

// Kernel to compute L2 norms for query vectors
template <int ThreadsPerBlock>
__global__ void computeQueryNormsKernel(
        const float* __restrict__ queries, // (n, d)
        float* __restrict__ norms,         // (n)
        idx_t n,
        int d) {
    idx_t idx = blockIdx.x * ThreadsPerBlock + threadIdx.x;
    if (idx >= n) return;

    const float* vec = queries + idx * d;
    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        float v = vec[i];
        sum += v * v;
    }
    norms[idx] = sum;
}

// Kernel to convert dot products to L2 distances and apply norms
// L2 distance = ||q||^2 - 2*q.c + ||c||^2
// Input dotProducts are already -2*q.c (from GEMM with alpha=-2)
template <int ThreadsPerBlock>
__global__ void addNormsToDistancesKernel(
        float* __restrict__ distances,        // (n, k_base), contains -2*q.c
        const float* __restrict__ queryNorms, // (n)
        const float* __restrict__ candNorms,  // (n * k_base)
        idx_t n,
        int k_base) {
    idx_t totalElements = n * k_base;
    for (idx_t idx = blockIdx.x * ThreadsPerBlock + threadIdx.x;
         idx < totalElements;
         idx += gridDim.x * ThreadsPerBlock) {
        idx_t queryIdx = idx / k_base;
        // distances[idx] already contains -2*q.c
        // Add ||q||^2 and ||c||^2
        float dist = distances[idx] + queryNorms[queryIdx] + candNorms[idx];
        // Clamp to zero (numerical stability)
        distances[idx] = dist > 0.0f ? dist : 0.0f;
    }
}

void runBatchedRerankL2Distance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& candidates,
        Tensor<float, 2, true>& outDistances,
        idx_t n,
        int k_base,
        int d) {
    FAISS_ASSERT(queries.getSize(0) == n);
    FAISS_ASSERT(queries.getSize(1) == d);
    FAISS_ASSERT(candidates.getSize(0) == n * k_base);
    FAISS_ASSERT(candidates.getSize(1) == d);
    FAISS_ASSERT(outDistances.getSize(0) == n);
    FAISS_ASSERT(outDistances.getSize(1) == k_base);

    // Step 1: Compute ||q||^2 for all queries
    DeviceTensor<float, 1, true> queryNorms(
            resources,
            makeTempAlloc(AllocType::Other, stream),
            {n});

    constexpr int kThreads = 256;
    int gridQ = utils::divUp(n, (idx_t)kThreads);
    computeQueryNormsKernel<kThreads><<<gridQ, kThreads, 0, stream>>>(
            queries.data(), queryNorms.data(), n, d);

    // Step 2: Compute ||c||^2 for all candidates
    DeviceTensor<float, 1, true> candNorms(
            resources,
            makeTempAlloc(AllocType::Other, stream),
            {n * k_base});

    int gridC = utils::divUp(n * k_base, (idx_t)kThreads);
    computeCandidateNormsKernel<kThreads><<<gridC, kThreads, 0, stream>>>(
            candidates.data(), candNorms.data(), n * k_base, d);

    // Step 3: Compute -2 * q . c using batched GEMM
    // We need to compute for each query i:
    //   queries[i] (1 x d) @ candidates[i*k_base:(i+1)*k_base] (k_base x d)^T = (1 x k_base)
    //
    // Using strided batched GEMM:
    //   A = queries reshaped as (n, 1, d), stride_A = d
    //   B = candidates reshaped as (n, k_base, d), stride_B = k_base * d
    //   C = outDistances reshaped as (n, 1, k_base), stride_C = k_base
    //
    // C_i = alpha * A_i @ B_i^T + beta * C_i
    // With alpha = -2.0, beta = 0.0

    auto handle = resources->getBlasHandleCurrentDevice();
    cublasSetStream(handle, stream);

    // cuBLAS uses column-major, so we need to be careful:
    // For row-major: C = A @ B^T with dimensions (1, k_base) = (1, d) @ (k_base, d)^T
    // In column-major: C^T = B @ A^T with dimensions (k_base, 1) = (k_base, d) @ (d, 1)
    //
    // So: m=k_base, n=1, k=d
    // transa=N (B is k_base x d, stored row-major = column-major with trans)
    // transb=T (A is 1 x d, stored row-major, need transpose)
    //
    // Actually simpler: treat as (k_base, d) @ (d, 1) = (k_base, 1)
    // B @ A^T where B=(k_base, d), A=(1, d)

    float alpha = -2.0f;
    float beta = 0.0f;

    cublasStatus_t err = cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,   // transa: transpose B (candidates)
            CUBLAS_OP_N,   // transb: no transpose A (queries)
            k_base,        // m: rows of op(B) and C
            1,             // n: cols of op(A) and C
            d,             // k: cols of op(B), rows of op(A)
            &alpha,
            candidates.data(),  // B: (n * k_base, d) row-major
            CUDA_R_32F,
            d,             // ldb: leading dimension of B
            (long long)(k_base * d),  // strideB: stride between batches
            queries.data(),     // A: (n, d) row-major
            CUDA_R_32F,
            d,             // lda: leading dimension of A
            (long long)d,       // strideA: stride between batches
            &beta,
            outDistances.data(), // C: (n, k_base) row-major, but we're writing (k_base, 1) per batch
            CUDA_R_32F,
            k_base,        // ldc: leading dimension of C
            (long long)k_base,  // strideC: stride between batches
            n,             // batch count
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);

    FAISS_ASSERT_FMT(err == CUBLAS_STATUS_SUCCESS,
            "cublasGemmStridedBatchedEx failed (%d)", (int)err);

    // Step 4: Add ||q||^2 + ||c||^2 to get final L2 distances
    int gridD = std::min(utils::divUp(n * k_base, (idx_t)kThreads), (idx_t)65536);
    addNormsToDistancesKernel<kThreads><<<gridD, kThreads, 0, stream>>>(
            outDistances.data(),
            queryNorms.data(),
            candNorms.data(),
            n,
            k_base);

    CUDA_TEST_ERROR();
}

void runBatchedRerankIPDistance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& candidates,
        Tensor<float, 2, true>& outDistances,
        idx_t n,
        int k_base,
        int d) {
    FAISS_ASSERT(queries.getSize(0) == n);
    FAISS_ASSERT(queries.getSize(1) == d);
    FAISS_ASSERT(candidates.getSize(0) == n * k_base);
    FAISS_ASSERT(candidates.getSize(1) == d);
    FAISS_ASSERT(outDistances.getSize(0) == n);
    FAISS_ASSERT(outDistances.getSize(1) == k_base);

    auto handle = resources->getBlasHandleCurrentDevice();
    cublasSetStream(handle, stream);

    // For IP, we just need q . c
    // Using same GEMM setup but with alpha = -1 (negative for distance ordering)
    float alpha = -1.0f;
    float beta = 0.0f;

    cublasStatus_t err = cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,   // transa
            CUBLAS_OP_N,   // transb
            k_base,        // m
            1,             // n
            d,             // k
            &alpha,
            candidates.data(),
            CUDA_R_32F,
            d,
            (long long)(k_base * d),
            queries.data(),
            CUDA_R_32F,
            d,
            (long long)d,
            &beta,
            outDistances.data(),
            CUDA_R_32F,
            k_base,
            (long long)k_base,
            n,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);

    FAISS_ASSERT_FMT(err == CUBLAS_STATUS_SUCCESS,
            "cublasGemmStridedBatchedEx failed (%d)", (int)err);

    CUDA_TEST_ERROR();
}

void runBatchedTopKRemap(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<float, 2, true>& distances,
        Tensor<idx_t, 2, true>& baseLabels,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outLabels,
        idx_t n,
        int k_base,
        int k,
        bool selectMin) {
    FAISS_ASSERT(distances.getSize(0) == n);
    FAISS_ASSERT(distances.getSize(1) == k_base);
    FAISS_ASSERT(baseLabels.getSize(0) == n);
    FAISS_ASSERT(baseLabels.getSize(1) == k_base);
    FAISS_ASSERT(outDistances.getSize(0) == n);
    FAISS_ASSERT(outDistances.getSize(1) == k);
    FAISS_ASSERT(outLabels.getSize(0) == n);
    FAISS_ASSERT(outLabels.getSize(1) == k);
    FAISS_ASSERT(k <= k_base);

    // Use faiss's existing runBlockSelectPair function
    // IMPORTANT: In faiss BlockSelect, dir=true selects MAXIMUM, dir=false selects MINIMUM
    // For L2 (selectMin=true), we need dir=false to select smallest distances
    // For IP (selectMin=false), we need dir=true to select largest (most negative = smallest IP)
    runBlockSelectPair(
            distances,
            baseLabels,
            outDistances,
            outLabels,
            !selectMin,  // dir: false for min (L2), true for max (IP)
            k,
            stream);

    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
