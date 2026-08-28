#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

// Reuse the pinned-host -> GPU-visible pointer translation used by the existing
// FreeToken expert-cache copy kernel (important on Windows/WDDM).
#include "fast_index_copy.cuh"

struct StridedIndexCopyParams {
    uint8_t* __restrict__ dst;
    const uint8_t* __restrict__ src;
    const void* __restrict__ dst_indices;
    const void* __restrict__ src_indices;
    const int64_t* __restrict__ valid_length;
    int64_t length;
    int64_t dst_stride_bytes;
    int64_t src_stride_bytes;
};

template <typename IdType, int kThreads, int kBlocks>
__global__ __launch_bounds__(kThreads) void strided_index_copy_kernel(
    const __grid_constant__ StridedIndexCopyParams p
) {
    const int64_t n = p.valid_length ? p.valid_length[0] : p.length;
    const int64_t units = p.src_stride_bytes >> 4;  // uint4 units
    const int64_t total = n * units;
    const auto* di = static_cast<const IdType*>(p.dst_indices);
    const auto* si = static_cast<const IdType*>(p.src_indices);
    const int64_t worker = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(kBlocks) * kThreads;

    for (int64_t u = worker; u < total; u += stride) {
        const int64_t item = u / units;
        const int64_t col = (u - item * units) << 4;
        const int64_t dst_row = static_cast<int64_t>(di[item]);
        const int64_t src_row = static_cast<int64_t>(si[item]);
        const uint4 value = *reinterpret_cast<const uint4*>(
            p.src + src_row * p.src_stride_bytes + col
        );
        *reinterpret_cast<uint4*>(
            p.dst + dst_row * p.dst_stride_bytes + col
        ) = value;
    }
}

template <int kThreads, int kBlocks>
struct StridedIndexCopyKernel {
    static void run(
        tvm::ffi::TensorView dst,
        tvm::ffi::TensorView dst_indices,
        tvm::ffi::TensorView src,
        tvm::ffi::TensorView src_indices,
        tvm::ffi::Optional<tvm::ffi::TensorView> num_indices
    ) {
        using namespace host;

        auto Dst = SymbolicSize{"destination stride"};
        auto Src = SymbolicSize{"source stride"};
        auto L = SymbolicSize{"indices length"};
        auto device = SymbolicDevice{};
        auto indices_dtype = SymbolicDType{};
        auto valid_dtype = SymbolicDType{};

        TensorMatcher({-1, Dst})
            .with_dtype<uint8_t>()
            .with_device<kDLCUDA>(device)
            .verify(dst);
        TensorMatcher({-1, Src})
            .with_dtype<uint8_t>()
            .with_device<kDLCUDAHost, kDLCPU, kDLCUDA>()
            .verify(src);
        TensorMatcher({L})
            .with_dtype<int32_t, int64_t>(indices_dtype)
            .with_device<kDLCUDA>(device)
            .verify(dst_indices)
            .verify(src_indices);

        const int64_t dst_stride = Dst.unwrap();
        const int64_t src_stride = Src.unwrap();
        RuntimeCheck(src_stride <= dst_stride,
            "strided_index_copy: source row is larger than destination slot");
        RuntimeCheck(src_stride > 0 && src_stride % 16 == 0,
            "strided_index_copy: source row bytes must be a positive multiple of 16");

        const int64_t* valid_length = nullptr;
        if (num_indices.has_value()) {
            TensorMatcher({1})
                .with_dtype<int64_t>(valid_dtype)
                .with_device<kDLCUDA>(device)
                .verify(num_indices.value());
            valid_length = static_cast<const int64_t*>(num_indices.value().data_ptr());
        }

        const auto params = StridedIndexCopyParams{
            static_cast<uint8_t*>(device_alias(dst.data_ptr(), dst.device())),
            static_cast<const uint8_t*>(device_alias(src.data_ptr(), src.device())),
            dst_indices.data_ptr(),
            src_indices.data_ptr(),
            valid_length,
            static_cast<int64_t>(L.unwrap()),
            dst_stride,
            src_stride,
        };
        const bool use_int32 = indices_dtype.unwrap().bits == 32;
        const auto kernel = use_int32
            ? strided_index_copy_kernel<int32_t, kThreads, kBlocks>
            : strided_index_copy_kernel<int64_t, kThreads, kBlocks>;
        LaunchKernel(kBlocks, kThreads, device.unwrap())(kernel, params);
    }
};
