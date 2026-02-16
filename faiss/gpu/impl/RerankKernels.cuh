/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/utils/Tensor.cuh>
#include <cuda_runtime.h>

namespace faiss {
namespace gpu {

class GpuResources;

/// Compute L2 distances between each query and its corresponding candidates.
/// For query i, computes distance from queries[i] to candidates[i*k_base : (i+1)*k_base]
///
/// @param queries (n, d) query vectors
/// @param candidates (n * k_base, d) candidate vectors (k_base candidates per query)
/// @param outDistances (n, k_base) output distance matrix
/// @param n number of queries
/// @param k_base number of candidates per query
/// @param d vector dimension
void runBatchedRerankL2Distance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& candidates,
        Tensor<float, 2, true>& outDistances,
        idx_t n,
        int k_base,
        int d);

/// Compute inner product distances between each query and its corresponding candidates.
/// For query i, computes IP from queries[i] to candidates[i*k_base : (i+1)*k_base]
///
/// @param queries (n, d) query vectors
/// @param candidates (n * k_base, d) candidate vectors (k_base candidates per query)
/// @param outDistances (n, k_base) output distance matrix
/// @param n number of queries
/// @param k_base number of candidates per query
/// @param d vector dimension
void runBatchedRerankIPDistance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& candidates,
        Tensor<float, 2, true>& outDistances,
        idx_t n,
        int k_base,
        int d);

/// Perform top-k selection on each row of the distance matrix and remap indices.
/// For each query, selects the k best distances from k_base candidates and remaps
/// the local indices (0 to k_base-1) to the original database indices using baseLabels.
///
/// @param distances (n, k_base) distance matrix
/// @param baseLabels (n, k_base) original database indices from base search
/// @param outDistances (n, k) output distances
/// @param outLabels (n, k) output indices (remapped to original database indices)
/// @param n number of queries
/// @param k_base number of candidates per query
/// @param k number of results to return per query
/// @param selectMin true for L2 (select minimum), false for IP (select maximum)
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
        bool selectMin);

} // namespace gpu
} // namespace faiss
