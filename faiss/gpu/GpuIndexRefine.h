/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <memory>

namespace faiss {

struct IndexRefine;

namespace gpu {

class FlatIndexSQ;

/// Storage type for refine index
enum class RefineStorageType {
    FLOAT32,    // GpuIndexFlat with float32
    FLOAT16,    // GpuIndexFlat with float16
    SQ8         // FlatIndexSQ with 8-bit scalar quantization
};

struct GpuIndexRefineConfig : public GpuIndexConfig {
    /// Oversampling factor: search k * k_factor candidates, refine to k
    float k_factor = 1.0f;

    /// Storage type for refine vectors (only used with SQ8 constructor)
    RefineStorageType storageType = RefineStorageType::FLOAT16;
};

/// GPU implementation of IndexRefine
/// Wraps a fast approximate GPU index (base) and an exact GPU index (refine)
/// to perform two-stage search: approximate search followed by exact refinement
class GpuIndexRefine : public GpuIndex {
   public:
    /// Construct from two GPU indexes (non-owning)
    /// @param provider GPU resources provider
    /// @param baseIndex Fast approximate search index (e.g., GpuIndexIVFPQ)
    /// @param refineIndex Exact search index for refinement (GpuIndexFlat)
    /// @param config Configuration options
    GpuIndexRefine(
            GpuResourcesProvider* provider,
            GpuIndex* baseIndex,
            GpuIndexFlat* refineIndex,
            GpuIndexRefineConfig config = GpuIndexRefineConfig());

    /// Construct from shared_ptr resources
    GpuIndexRefine(
            std::shared_ptr<GpuResources> resources,
            GpuIndex* baseIndex,
            GpuIndexFlat* refineIndex,
            GpuIndexRefineConfig config = GpuIndexRefineConfig());

    /// Construct with SQ8 storage for refine (memory efficient)
    /// This creates an internal FlatIndexSQ for refine storage
    GpuIndexRefine(
            GpuResourcesProvider* provider,
            GpuIndex* baseIndex,
            GpuIndexRefineConfig config);

    /// Construct with SQ8 storage from shared_ptr resources
    GpuIndexRefine(
            std::shared_ptr<GpuResources> resources,
            GpuIndex* baseIndex,
            GpuIndexRefineConfig config);

    ~GpuIndexRefine() override;

    /// Reset both base and refine indexes
    void reset() override;

    /// Train both indexes
    void train(idx_t n, const float* x) override;

    /// Set k_factor (oversampling ratio)
    void setKFactor(float k);

    /// Get current k_factor
    float getKFactor() const;

    /// Access the base index
    GpuIndex* getBaseIndex() {
        return baseIndex_;
    }

    /// Access the refine index
    GpuIndexFlat* getRefineIndex() {
        return refineIndex_;
    }

   protected:
    /// Does not require IDs (delegates to base)
    bool addImplRequiresIDs_() const override;

    /// Add vectors to both indexes
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    /// Two-stage search: base search + GPU refinement
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   private:
    /// Initialize from given indexes
    void init_(
            GpuIndex* baseIndex,
            GpuIndexFlat* refineIndex,
            GpuIndexRefineConfig config);

    /// Initialize with SQ8 storage
    void initSQ_(
            GpuIndex* baseIndex,
            GpuIndexRefineConfig config);

    /// Fast approximate search index
    GpuIndex* baseIndex_;

    /// Exact refinement index (used when storageType != SQ8)
    GpuIndexFlat* refineIndex_;

    /// SQ8 refinement storage (used when storageType == SQ8)
    std::unique_ptr<FlatIndexSQ> refineIndexSQ_;

    /// Whether we own the base index
    bool ownBaseIndex_;

    /// Whether we own the refine index
    bool ownRefineIndex_;

    /// Oversampling factor for search
    float k_factor_;

    /// Configuration
    GpuIndexRefineConfig config_;

    /// Whether using SQ8 storage
    bool useSQ_;
};

} // namespace gpu
} // namespace faiss
