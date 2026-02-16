/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>

namespace faiss {

struct IndexRefine;

namespace gpu {

struct GpuIndexRefineConfig : public GpuIndexConfig {
    /// Oversampling factor: search k * k_factor candidates, refine to k
    float k_factor = 1.0f;
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

    /// Fast approximate search index
    GpuIndex* baseIndex_;

    /// Exact refinement index
    GpuIndexFlat* refineIndex_;

    /// Whether we own the base index
    bool ownBaseIndex_;

    /// Whether we own the refine index
    bool ownRefineIndex_;

    /// Oversampling factor for search
    float k_factor_;

    /// Configuration
    GpuIndexRefineConfig config_;
};

} // namespace gpu
} // namespace faiss
