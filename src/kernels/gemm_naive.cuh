#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gemm_precision.cuh"

namespace gemm_foundry::detail {

template <typename TA, typename TB, typename TC, typename AccT>
__global__ void gemm_naive_kernel(
    const TA* A,
    const TB* B,
    TC* C,
    std::int64_t M,
    std::int64_t N,
    std::int64_t K,
    AccT alpha,
    AccT beta,
    AccT scale_a ,
    AccT scale_b,
    std::int64_t lda,
    std::int64_t ldb,
    std::int64_t ldc,
    bool transpose_a,
    bool transpose_b,
    std::int64_t stride_a,
    std::int64_t stride_b,
    std::int64_t stride_c
) {
    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::int64_t batch =static_cast<std::int64_t>(blockIdx.z);

    if (row >= M || col >= N)return;

    const std::int64_t a_base = batch * stride_a;
    const std::int64_t b_base = batch * stride_b;
    const std::int64_t c_base = batch * stride_c;

    AccT sum = AccT{0};

    // const AccT alpha_acc = to_accum(alpha);
    // const AccT beta_acc  = to_accum(beta);

    for (std::int64_t k = 0; k < K; ++k) {
        const std::int64_t a_idx = transpose_a ? (a_base + k * lda + row) : (a_base + row * lda + k);
        const std::int64_t b_idx = transpose_b ? (b_base + col * ldb + k) : (b_base + k * ldb + col);

        sum += to_accum(A[a_idx]) * to_accum(B[b_idx]);
    }

    if (row < M && col < N) {
        const std::int64_t c_idx=c_base + row * ldc + col;

        const AccT scaled_sum = scale_a * scale_b * sum;

        AccT result = alpha * scaled_sum;
        if (beta != AccT{0}) {
            result += beta * to_accum(C[c_idx]);
        }
        C[c_idx] = from_accum<TC>(result);
    }
}

} // namespace gemm_foundry::detail
