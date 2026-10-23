#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gemm_precision.cuh"

namespace gemm_foundry::detail {

template <typename TA, typename TB, typename TC, typename AccT>
__global__ void gemm_tiled_kernel(
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
    constexpr int TILE = 16;
    const int tx= threadIdx.x;
    const int ty =threadIdx.y;

    const std::int64_t col = static_cast<std::int64_t>(blockIdx.x) * TILE + tx;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.y) * TILE + ty;
    const std::int64_t batch=static_cast<std::int64_t>(blockIdx.z);

    const std::int64_t a_base =batch * stride_a;
    const std::int64_t b_base= batch *stride_b;
    const std::int64_t c_base= batch* stride_c;

    __shared__ TA tile_a[TILE][TILE];
    __shared__ TB tile_b[TILE][TILE];

    using AccT = AccumulationTypeT<T>;
    AccT sum = AccT{0};

    // Process the K dimension in chunks of 16.
    for (std::int64_t base_k = 0;base_k < K;base_k += TILE) {
        const std::int64_t a_col = base_k + tx;
        const std::int64_t b_row = base_k + ty;
        // Cooperatively load A's tile
        if (row < M && a_col < K) {
            const std::int64_t a_idx = transpose_a ? (a_base + a_col * lda + row) : (a_base + row * lda + a_col);
            tile_a[ty][tx] = A[a_idx];
        } else {
            tile_a[ty][tx] = storage_zero<TA>();
        }

        // Cooperatively load B's tile
        if (b_row < K && col < N) {
            const std::int64_t b_idx = transpose_b ? (b_base + col * ldb + b_row) : (b_base + b_row * ldb + col);
            tile_b[ty][tx] = B[b_idx];
        } else {
            tile_b[ty][tx] = storage_zero<TB>();
        }

        __syncthreads();

        // partial dot product
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += to_accum(tile_a[ty][k]) * to_accum(tile_b[k][tx]);
        }

        // Finish reading before the tiles are overwritten
        __syncthreads();
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
