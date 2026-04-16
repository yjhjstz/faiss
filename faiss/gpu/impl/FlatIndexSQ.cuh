/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <memory>

namespace faiss {
namespace gpu {

/// GPU storage for SQ-encoded vectors
/// Supports 8-bit (QT_8bit / QT_8bit_uniform) and
/// 4-bit (QT_4bit / QT_4bit_uniform) scalar quantized formats.
class FlatIndexSQ {
   public:
    FlatIndexSQ(
            GpuResources* res,
            int dim,
            ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QT_8bit,
            MemorySpace space = MemorySpace::Device);

    ~FlatIndexSQ();

    /// Train the scalar quantizer
    void train(idx_t n, const float* x);

    /// Returns whether we are trained
    bool isTrained() const { return trained_; }

    /// Add vectors (will be SQ encoded)
    void add(idx_t n, const float* x, cudaStream_t stream);

    /// Get number of vectors
    idx_t getSize() const { return numVecs_; }

    /// Get dimension
    int getDim() const { return dim_; }

    /// Reset the index
    void reset();

    /// Reconstruct vectors by indices to GPU memory
    /// keys: device tensor of indices
    /// out: device tensor for reconstructed vectors (n, dim)
    void reconstruct(
            Tensor<idx_t, 1, true>& keys,
            Tensor<float, 2, true>& out,
            cudaStream_t stream);

    /// Get SQ parameters for distance computation
    const ScalarQuantizer& getScalarQuantizer() const { return *sq_; }

    /// Get raw encoded data pointer
    uint8_t* getCodesData() { return codes_.data(); }

    /// Get trained parameters on GPU
    float* getVminData() { return vmin_.data(); }
    float* getVdiffData() { return vdiff_.data(); }

   private:
    GpuResources* resources_;
    int dim_;
    ScalarQuantizer::QuantizerType qtype_;
    /// Bytes per encoded vector: dim for 8-bit, (dim+1)/2 for 4-bit.
    int codeSize_;
    MemorySpace space_;
    bool trained_;
    idx_t numVecs_;

    /// SQ parameters (CPU)
    std::unique_ptr<ScalarQuantizer> sq_;

    /// SQ parameters on GPU: vmin and vdiff per dimension
    DeviceVector<float> vmin_;
    DeviceVector<float> vdiff_;

    /// Encoded vectors (numVecs * codeSize_ bytes)
    DeviceVector<uint8_t> codes_;
};

} // namespace gpu
} // namespace faiss
