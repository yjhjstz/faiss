/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/FlatIndexSQ.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace faiss {
namespace gpu {

// Kernel to encode float vectors to SQ8
__global__ void encodeSQ8Kernel(
        const float* __restrict__ input,  // (n, d)
        uint8_t* __restrict__ output,     // (n, d)
        const float* __restrict__ vmin,   // (d)
        const float* __restrict__ vdiff,  // (d)
        idx_t n,
        int d) {
    idx_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx_t total = n * d;

    for (; idx < total; idx += gridDim.x * blockDim.x) {
        int dim = idx % d;
        float v = input[idx];

        // Encode: (v - vmin) / vdiff * 255
        float x = (v - vmin[dim]) / vdiff[dim];
        x = fminf(1.0f, fmaxf(0.0f, x));
        output[idx] = (uint8_t)(x * 255.0f);
    }
}

// Kernel to encode float vectors to SQ4 (two 4-bit codes packed per byte)
__global__ void encodeSQ4Kernel(
        const float* __restrict__ input,  // (n, d)
        uint8_t* __restrict__ output,     // (n, codeSize)
        const float* __restrict__ vmin,   // (d)
        const float* __restrict__ vdiff,  // (d)
        idx_t n,
        int d,
        int codeSize) {
    idx_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx_t total = n * (idx_t)codeSize;

    for (; idx < total; idx += gridDim.x * blockDim.x) {
        idx_t row = idx / codeSize;
        int col = idx - row * codeSize; // byte index within row
        int dim0 = col * 2;
        int dim1 = dim0 + 1;

        float v0 = input[row * d + dim0];
        float x0 = (v0 - vmin[dim0]) / vdiff[dim0];
        x0 = fminf(1.0f, fmaxf(0.0f, x0));
        uint8_t c0 = (uint8_t)(x0 * 15.0f);

        uint8_t c1 = 0;
        if (dim1 < d) {
            float v1 = input[row * d + dim1];
            float x1 = (v1 - vmin[dim1]) / vdiff[dim1];
            x1 = fminf(1.0f, fmaxf(0.0f, x1));
            c1 = (uint8_t)(x1 * 15.0f);
        }

        output[idx] = (uint8_t)((c0 & 0xf) | ((c1 & 0xf) << 4));
    }
}

// Kernel to decode SQ8 vectors to float, with index gather
__global__ void decodeSQ8GatherKernel(
        const uint8_t* __restrict__ codes,  // (total_vecs, d)
        const idx_t* __restrict__ indices,  // (n)
        float* __restrict__ output,         // (n, d)
        const float* __restrict__ vmin,     // (d)
        const float* __restrict__ vdiff,    // (d)
        idx_t n,
        int d,
        idx_t numVecs) {
    idx_t row = blockIdx.x;
    if (row >= n) return;

    idx_t srcIdx = indices[row];
    // Invalid index: negative or out of bounds
    if (srcIdx < 0 || srcIdx >= numVecs) {
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            output[row * d + i] = 0.0f;
        }
        return;
    }

    const uint8_t* srcRow = codes + srcIdx * d;
    float* dstRow = output + row * d;

    // Decode: vmin + (code + 0.5) / 255 * vdiff
    // Simplified: vmin' + code * vdiff' where vdiff' = vdiff/255, vmin' = vmin + 0.5*vdiff'
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float vd = vdiff[i] / 255.0f;
        float vm = vmin[i] + 0.5f * vd;
        dstRow[i] = vm + (float)srcRow[i] * vd;
    }
}

// Kernel to decode SQ4 vectors to float, with index gather
__global__ void decodeSQ4GatherKernel(
        const uint8_t* __restrict__ codes,  // (total_vecs, codeSize)
        const idx_t* __restrict__ indices,  // (n)
        float* __restrict__ output,         // (n, d)
        const float* __restrict__ vmin,     // (d)
        const float* __restrict__ vdiff,    // (d)
        idx_t n,
        int d,
        int codeSize,
        idx_t numVecs) {
    idx_t row = blockIdx.x;
    if (row >= n) return;

    idx_t srcIdx = indices[row];
    if (srcIdx < 0 || srcIdx >= numVecs) {
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            output[row * d + i] = 0.0f;
        }
        return;
    }

    const uint8_t* srcRow = codes + srcIdx * (idx_t)codeSize;
    float* dstRow = output + row * d;

    // Decode: vmin + (code + 0.5) / 15 * vdiff
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        uint8_t byte = srcRow[i >> 1];
        uint8_t code = (byte >> ((i & 1) << 2)) & 0xf;
        float vd = vdiff[i] / 15.0f;
        float vm = vmin[i] + 0.5f * vd;
        dstRow[i] = vm + (float)code * vd;
    }
}

FlatIndexSQ::FlatIndexSQ(
        GpuResources* res,
        int dim,
        ScalarQuantizer::QuantizerType qtype,
        MemorySpace space)
        : resources_(res),
          dim_(dim),
          qtype_(qtype),
          codeSize_(
                  (qtype == ScalarQuantizer::QT_4bit ||
                   qtype == ScalarQuantizer::QT_4bit_uniform)
                          ? (dim + 1) / 2
                          : dim),
          space_(space),
          trained_(false),
          numVecs_(0),
          sq_(std::make_unique<ScalarQuantizer>(dim, qtype)),
          vmin_(res, AllocInfo(AllocType::Quantizer, 0, space, res->getDefaultStreamCurrentDevice())),
          vdiff_(res, AllocInfo(AllocType::Quantizer, 0, space, res->getDefaultStreamCurrentDevice())),
          codes_(res, AllocInfo(AllocType::Other, 0, space, res->getDefaultStreamCurrentDevice())) {
    FAISS_THROW_IF_NOT_MSG(
            qtype == ScalarQuantizer::QT_8bit ||
            qtype == ScalarQuantizer::QT_8bit_uniform ||
            qtype == ScalarQuantizer::QT_4bit ||
            qtype == ScalarQuantizer::QT_4bit_uniform,
            "FlatIndexSQ supports QT_8bit, QT_8bit_uniform, "
            "QT_4bit and QT_4bit_uniform");
}

FlatIndexSQ::~FlatIndexSQ() = default;

void FlatIndexSQ::train(idx_t n, const float* x) {
    // Train the CPU scalar quantizer
    sq_->train(n, x);
    trained_ = true;

    // Copy trained parameters to GPU
    auto stream = resources_->getDefaultStreamCurrentDevice();

    bool isUniform =
            (qtype_ == ScalarQuantizer::QT_8bit_uniform ||
             qtype_ == ScalarQuantizer::QT_4bit_uniform);

    if (!isUniform) {
        // Per-dimension min/diff
        vmin_.resize(dim_, stream);
        vdiff_.resize(dim_, stream);

        // sq_->trained contains [vmin[0], vmin[1], ..., vmin[d-1], vdiff[0], ...]
        std::vector<float> vminHost(dim_);
        std::vector<float> vdiffHost(dim_);
        for (int i = 0; i < dim_; i++) {
            vminHost[i] = sq_->trained[i];
            vdiffHost[i] = sq_->trained[dim_ + i];
        }

        CUDA_VERIFY(cudaMemcpyAsync(
                vmin_.data(), vminHost.data(),
                dim_ * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_VERIFY(cudaMemcpyAsync(
                vdiff_.data(), vdiffHost.data(),
                dim_ * sizeof(float), cudaMemcpyHostToDevice, stream));
    } else {
        // Uniform: single min/diff for all dimensions
        vmin_.resize(1, stream);
        vdiff_.resize(1, stream);

        CUDA_VERIFY(cudaMemcpyAsync(
                vmin_.data(), &sq_->trained[0],
                sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_VERIFY(cudaMemcpyAsync(
                vdiff_.data(), &sq_->trained[1],
                sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    CUDA_VERIFY(cudaStreamSynchronize(stream));
}

void FlatIndexSQ::add(idx_t n, const float* x, cudaStream_t stream) {
    FAISS_THROW_IF_NOT_MSG(trained_, "FlatIndexSQ must be trained before adding");

    if (n == 0) return;

    bool is4bit =
            (qtype_ == ScalarQuantizer::QT_4bit ||
             qtype_ == ScalarQuantizer::QT_4bit_uniform);
    bool isUniform =
            (qtype_ == ScalarQuantizer::QT_8bit_uniform ||
             qtype_ == ScalarQuantizer::QT_4bit_uniform);

    // Resize codes buffer (in bytes)
    idx_t newSize = numVecs_ + n;
    codes_.resize(newSize * (idx_t)codeSize_, stream);

    // Copy input to device
    DeviceTensor<float, 2, true> inputDevice(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {n, dim_});
    CUDA_VERIFY(cudaMemcpyAsync(
            inputDevice.data(),
            x,
            n * dim_ * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));

    // Broadcast vmin/vdiff so we can always index per-dimension in kernels
    DeviceTensor<float, 1, true> vminBcast(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {dim_});
    DeviceTensor<float, 1, true> vdiffBcast(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {dim_});

    if (!isUniform) {
        CUDA_VERIFY(cudaMemcpyAsync(
                vminBcast.data(), vmin_.data(),
                dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_VERIFY(cudaMemcpyAsync(
                vdiffBcast.data(), vdiff_.data(),
                dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else {
        // Broadcast uniform values
        float vminVal, vdiffVal;
        CUDA_VERIFY(cudaMemcpyAsync(
                &vminVal, vmin_.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_VERIFY(cudaMemcpyAsync(
                &vdiffVal, vdiff_.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_VERIFY(cudaStreamSynchronize(stream));

        thrust::fill(thrust::cuda::par.on(stream),
                     vminBcast.data(), vminBcast.data() + dim_, vminVal);
        thrust::fill(thrust::cuda::par.on(stream),
                     vdiffBcast.data(), vdiffBcast.data() + dim_, vdiffVal);
    }

    int threads = 256;
    if (is4bit) {
        int blocks = std::min(
                (int)utils::divUp(n * (idx_t)codeSize_, (idx_t)threads),
                65535);
        encodeSQ4Kernel<<<blocks, threads, 0, stream>>>(
                inputDevice.data(),
                codes_.data() + numVecs_ * (idx_t)codeSize_,
                vminBcast.data(),
                vdiffBcast.data(),
                n,
                dim_,
                codeSize_);
    } else {
        int blocks = std::min(
                (int)utils::divUp(n * (idx_t)dim_, (idx_t)threads), 65535);
        encodeSQ8Kernel<<<blocks, threads, 0, stream>>>(
                inputDevice.data(),
                codes_.data() + numVecs_ * (idx_t)codeSize_,
                vminBcast.data(),
                vdiffBcast.data(),
                n,
                dim_);
    }

    CUDA_TEST_ERROR();
    numVecs_ = newSize;
}

void FlatIndexSQ::reset() {
    numVecs_ = 0;
    codes_.clear();
}

void FlatIndexSQ::reconstruct(
        Tensor<idx_t, 1, true>& keys,
        Tensor<float, 2, true>& out,
        cudaStream_t stream) {
    FAISS_THROW_IF_NOT_MSG(trained_, "FlatIndexSQ must be trained before reconstruct");

    idx_t n = keys.getSize(0);
    FAISS_ASSERT(out.getSize(0) == n);
    FAISS_ASSERT(out.getSize(1) == dim_);

    if (n == 0) return;

    // Prepare vmin/vdiff for kernel
    DeviceTensor<float, 1, true> vminBcast(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {dim_});
    DeviceTensor<float, 1, true> vdiffBcast(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {dim_});

    bool is4bit =
            (qtype_ == ScalarQuantizer::QT_4bit ||
             qtype_ == ScalarQuantizer::QT_4bit_uniform);
    bool isUniform =
            (qtype_ == ScalarQuantizer::QT_8bit_uniform ||
             qtype_ == ScalarQuantizer::QT_4bit_uniform);

    if (!isUniform) {
        CUDA_VERIFY(cudaMemcpyAsync(
                vminBcast.data(), vmin_.data(),
                dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_VERIFY(cudaMemcpyAsync(
                vdiffBcast.data(), vdiff_.data(),
                dim_ * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else {
        float vminVal, vdiffVal;
        CUDA_VERIFY(cudaMemcpyAsync(
                &vminVal, vmin_.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_VERIFY(cudaMemcpyAsync(
                &vdiffVal, vdiff_.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_VERIFY(cudaStreamSynchronize(stream));

        thrust::fill(thrust::cuda::par.on(stream),
                     vminBcast.data(), vminBcast.data() + dim_, vminVal);
        thrust::fill(thrust::cuda::par.on(stream),
                     vdiffBcast.data(), vdiffBcast.data() + dim_, vdiffVal);
    }

    int threads = 256;

    if (is4bit) {
        decodeSQ4GatherKernel<<<n, threads, 0, stream>>>(
                codes_.data(),
                keys.data(),
                out.data(),
                vminBcast.data(),
                vdiffBcast.data(),
                n,
                dim_,
                codeSize_,
                numVecs_);
    } else {
        decodeSQ8GatherKernel<<<n, threads, 0, stream>>>(
                codes_.data(),
                keys.data(),
                out.data(),
                vminBcast.data(),
                vdiffBcast.data(),
                n,
                dim_,
                numVecs_);
    }

    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
