#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "dispatch.h"
#include "ggml-common.h"
#include "vecdotq.cuh"

// Variant of the vendored llama.cpp/vLLM GGUF MoE MMVQ path that accepts an
// explicit byte stride between expert slots. FreeToken's Unsloth Dynamic cache
// pads every slot to the largest native expert row for that projection, while
// each layer keeps its original GGML quant type and packed byte count.

template <typename scalar_t>
static __global__ void quantize_q8_1_strided(
    const scalar_t* __restrict__ x,
    void* __restrict__ vy,
    const int kx,
    const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1* y = (block_q8_1*)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;
  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), amax, mask, 32));
    sum += SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), sum, mask, 32);
  }
  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
  y[ib].qs[iqs] = q;
  if (iqs == 0) {
    y[ib].ds.x = __float2half(d);
    y[ib].ds.y = __float2half(sum);
  }
}

template <typename scalar_t>
static void quantize_row_q8_1_strided_cuda(
    const scalar_t* x,
    void* vy,
    const int kx,
    const int ky,
    cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1_strided<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx],
        (int32_t*)vy + off * (kx_padded / 32 * 9),
        kx,
        kx_padded);
  }
}

template <
    typename scalar_t,
    int qk,
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void moe_vec_q_strided(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* topk_ids,
    const int topk,
    const int ncols,
    const int nrows,
    const int token_stride,
    const int64_t expert_stride_bytes) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;
  const auto token = blockIdx.z / topk;
  const auto expert = topk_ids[blockIdx.z];
  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;
  const auto* expert_base =
      static_cast<const uint8_t*>(vx) + static_cast<int64_t>(expert) * expert_stride_bytes;
  const auto* x = reinterpret_cast<const block_q_t*>(expert_base);
  const auto* y =
      (const block_q8_1*)(((const int*)vy) + token * token_stride);

  float tmp = 0.0f;
  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i;
    const int iby = i * (qk / QK8_1);
    const int iqs = vdr * (threadIdx.x % (qi / vdr));
    tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
  }
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    tmp += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp, mask);
  }
  if (threadIdx.x == 0) {
    dst[blockIdx.z * nrows + row] = tmp;
  }
}

template <
    typename scalar_t,
    int qk,
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_cuda_t vec_dot_q_cuda>
static void launch_moe_vec_strided(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int* topk_ids,
    const int top_k,
    const int tokens,
    const int ncols,
    const int nrows,
    const int token_stride,
    const int64_t expert_stride_bytes,
    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q_strided<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>
      <<<block_nums, block_dims, 0, stream>>>(
          vx,
          vy,
          dst,
          topk_ids,
          top_k,
          ncols,
          nrows,
          token_stride,
          expert_stride_bytes);
}

torch::Tensor ggml_moe_a8_vec_strided(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor topk_ids,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens,
    int64_t expert_stride_bytes) {
  const int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::zeros({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);

  TORCH_CHECK(W.scalar_type() == torch::kUInt8, "GGUF strided MoE weights must be uint8");
  TORCH_CHECK(W.is_cuda(), "GGUF strided MoE weights must be on CUDA");
  TORCH_CHECK(expert_stride_bytes > 0, "expert_stride_bytes must be positive");

  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_vec_a8_strided", [&] {
    quantize_row_q8_1_strided_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
#define RUN(QK, QI, BLOCK, VDR, DOT) \
    launch_moe_vec_strided<scalar_t, QK, QI, BLOCK, VDR, DOT>( \
        (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), \
        (int*)topk_ids.data_ptr(), top_k, tokens, col, row, quant_X.stride(0), \
        expert_stride_bytes, stream)
    switch (type) {
      case 2:  RUN(QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1); break;
      case 3:  RUN(QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1); break;
      case 6:  RUN(QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1); break;
      case 7:  RUN(QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1); break;
      case 8:  RUN(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1); break;
      case 10: RUN(QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1); break;
      case 11: RUN(QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1); break;
      case 12: RUN(QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1); break;
      case 13: RUN(QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1); break;
      case 14: RUN(QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1); break;
      case 16: RUN(QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1); break;
      case 17: RUN(QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1); break;
      case 18: RUN(QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1); break;
      case 19: RUN(QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1); break;
      case 20: RUN(QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1); break;
      case 21: RUN(QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1); break;
      case 22: RUN(QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1); break;
      case 23: RUN(QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1); break;
      case 29: RUN(QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1); break;
      default:
        TORCH_CHECK(false, "unsupported GGUF MoE quant type for strided cache: ", type);
    }
#undef RUN
  });
  return Y;
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_moe_a8_vec_strided", &ggml_moe_a8_vec_strided, "");
}
