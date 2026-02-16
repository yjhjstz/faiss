/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndexRefine.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>

namespace faiss {
namespace gpu {

GpuIndexRefine::GpuIndexRefine(
        GpuResourcesProvider* provider,
        GpuIndex* baseIndex,
        GpuIndexFlat* refineIndex,
        GpuIndexRefineConfig config)
        : GpuIndex(
                  provider->getResources(),
                  baseIndex->d,
                  baseIndex->metric_type,
                  baseIndex->metric_arg,
                  config) {
    init_(baseIndex, refineIndex, config);
}

GpuIndexRefine::GpuIndexRefine(
        std::shared_ptr<GpuResources> resources,
        GpuIndex* baseIndex,
        GpuIndexFlat* refineIndex,
        GpuIndexRefineConfig config)
        : GpuIndex(
                  resources,
                  baseIndex->d,
                  baseIndex->metric_type,
                  baseIndex->metric_arg,
                  config) {
    init_(baseIndex, refineIndex, config);
}

void GpuIndexRefine::init_(
        GpuIndex* baseIndex,
        GpuIndexFlat* refineIndex,
        GpuIndexRefineConfig config) {
    FAISS_THROW_IF_NOT(baseIndex);
    FAISS_THROW_IF_NOT(refineIndex);

    // Validate that dimensions and metrics match
    FAISS_THROW_IF_NOT_MSG(
            baseIndex->d == refineIndex->d,
            "GpuIndexRefine: base and refine indexes must have same dimension");
    FAISS_THROW_IF_NOT_MSG(
            baseIndex->metric_type == refineIndex->metric_type,
            "GpuIndexRefine: base and refine indexes must have same metric");

    baseIndex_ = baseIndex;
    refineIndex_ = refineIndex;
    ownBaseIndex_ = false;
    ownRefineIndex_ = false;
    k_factor_ = config.k_factor;
    config_ = config;

    // Copy ntotal from base index
    this->ntotal = baseIndex->ntotal;
    this->is_trained = baseIndex->is_trained;
}

GpuIndexRefine::~GpuIndexRefine() {
    if (ownBaseIndex_ && baseIndex_) {
        delete baseIndex_;
    }
    if (ownRefineIndex_ && refineIndex_) {
        delete refineIndex_;
    }
}

void GpuIndexRefine::reset() {
    DeviceScope scope(config_.device);

    if (baseIndex_) {
        baseIndex_->reset();
    }
    if (refineIndex_) {
        refineIndex_->reset();
    }
    this->ntotal = 0;
}

void GpuIndexRefine::train(idx_t n, const float* x) {
    if (baseIndex_) {
        baseIndex_->train(n, x);
    }
    if (refineIndex_) {
        refineIndex_->train(n, x);
    }
    this->is_trained = true;
}

void GpuIndexRefine::setKFactor(float k) {
    FAISS_THROW_IF_NOT_MSG(k >= 1.0f, "k_factor must be >= 1.0");
    k_factor_ = k;
}

float GpuIndexRefine::getKFactor() const {
    return k_factor_;
}

bool GpuIndexRefine::addImplRequiresIDs_() const {
    return false;
}

void GpuIndexRefine::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    // Add to both indexes
    FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported for GpuIndexRefine");

    if (baseIndex_) {
        baseIndex_->add(n, x);
    }
    if (refineIndex_) {
        refineIndex_->add(n, x);
    }
    this->ntotal += n;
}

void GpuIndexRefine::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(baseIndex_);
    FAISS_THROW_IF_NOT(refineIndex_);

    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    // Calculate k_base with oversampling
    int k_base = std::max(k, static_cast<int>(std::ceil(k * k_factor_)));

    // Ensure k_base doesn't exceed ntotal
    k_base = std::min(k_base, static_cast<int>(ntotal));

    if (k_base == 0 || n == 0) {
        return;
    }

    // Allocate temporary buffers for base search results
    DeviceTensor<float, 2, true> baseDistances(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n, k_base});
    DeviceTensor<idx_t, 2, true> baseLabels(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n, k_base});

    // Step 1: Search base index for k_base candidates
    baseIndex_->search(
            n, x, k_base,
            baseDistances.data(), baseLabels.data(), params);

    // Step 2: Gather candidate vectors from refine index
    DeviceTensor<float, 2, true> candidates(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n * k_base, this->d});

    FlatIndex* flatData = refineIndex_->getGpuData();
    FAISS_THROW_IF_NOT(flatData);

    // Reshape baseLabels to 1D for reconstruct call
    Tensor<idx_t, 1, true> baseLabels1D = baseLabels.view<1>({n * k_base});
    flatData->reconstruct(baseLabels1D, candidates);

    // Step 3: For each query, compute exact distances and select top-k
    // Copy queries to device
    auto queriesDevice = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(x),
            stream,
            {n, this->d});

    // Allocate output tensors on device
    DeviceTensor<float, 2, true> outDistancesDevice(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n, k});
    DeviceTensor<idx_t, 2, true> outLabelsDevice(
            resources_.get(),
            makeTempAlloc(AllocType::Other, stream),
            {n, k});

    // Process each query: compute distances to its candidates and select top-k
    for (idx_t i = 0; i < n; i++) {
        // Get query i
        Tensor<float, 2, true> query = queriesDevice.narrowOutermost(i, 1);

        // Get candidates for query i
        Tensor<float, 2, true> queryCandidates =
                candidates.narrowOutermost(i * k_base, k_base);

        // Temporary output for this query
        DeviceTensor<float, 2, true> queryOutDist(
                resources_.get(),
                makeTempAlloc(AllocType::Other, stream),
                {1, k});
        DeviceTensor<idx_t, 2, true> queryOutIdx(
                resources_.get(),
                makeTempAlloc(AllocType::Other, stream),
                {1, k});

        // Use bfKnnOnDevice to compute top-k from the candidates
        // This computes distances and selects top-k in one operation
        bfKnnOnDevice<float>(
                resources_.get(),
                config_.device,
                stream,
                queryCandidates,  // database: candidates for this query
                true,  // row major
                nullptr,  // no precomputed norms
                query,  // query
                true,  // row major
                k,  // we want top-k
                metric_type,
                metric_arg,
                queryOutDist,
                queryOutIdx,
                false);  // don't ignore distances

        // The indices from bfKnnOnDevice are local (0 to k_base-1)
        // We need to map them back to the original database indices using baseLabels
        // Copy results to host, remap indices, and copy back

        // Copy local indices to host
        std::vector<idx_t> localIdx(k);
        fromDevice(queryOutIdx.data(), localIdx.data(), k, stream);
        cudaStreamSynchronize(stream);

        // Copy baseLabels for this query to host
        std::vector<idx_t> baseLab(k_base);
        fromDevice(baseLabels.data() + i * k_base, baseLab.data(), k_base, stream);
        cudaStreamSynchronize(stream);

        // Remap indices
        std::vector<idx_t> remappedIdx(k);
        for (int j = 0; j < k; j++) {
            idx_t localIndex = localIdx[j];
            if (localIndex >= 0 && localIndex < k_base) {
                remappedIdx[j] = baseLab[localIndex];
            } else {
                remappedIdx[j] = -1;
            }
        }

        // Copy remapped indices back to device output
        cudaMemcpyAsync(
                outLabelsDevice.data() + i * k,
                remappedIdx.data(),
                k * sizeof(idx_t),
                cudaMemcpyHostToDevice,
                stream);

        // Copy distances to output
        cudaMemcpyAsync(
                outDistancesDevice.data() + i * k,
                queryOutDist.data(),
                k * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream);
    }

    // Copy final results to host output
    fromDevice<float, 2>(outDistancesDevice, distances, stream);
    fromDevice<idx_t, 2>(outLabelsDevice, labels, stream);

    cudaStreamSynchronize(stream);
}

} // namespace gpu
} // namespace faiss
