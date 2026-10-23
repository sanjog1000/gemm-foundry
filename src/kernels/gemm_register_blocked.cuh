#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gemm_precision.cuh"

namespace gemm_foundry::detail {

template <typename TA, typename TB, typename TC, typename AccT>
__global__ void gemm_register_blocked_kernel(
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
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const std::int64_t batch =static_cast<std::int64_t>(blockIdx.z);
    const std::int64_t row_base =static_cast<std::int64_t>(blockIdx.y) * BM;
    const std::int64_t col =static_cast<std::int64_t>(blockIdx.x) * BN + tx;

    const std::int64_t a_base = batch * stride_a;
    const std::int64_t b_base = batch * stride_b;
    const std::int64_t c_base = batch * stride_c;

    __shared__ TA tile_a[BM][BK];
    __shared__ TB tile_b[BK][BN];

    // Four output accumulators per thread.
    AccT accum[4] = {AccT(0), AccT(0), AccT(0), AccT(0)};

    for (std::int64_t base_k = 0;base_k < K;base_k += BK) {

        // Cooperatively load A.
        for (int q = 0; q < 4; ++q) {
            const int tile_row = ty + q * 8;

            const std::int64_t global_row =row_base + tile_row;
            const std::int64_t global_k =base_k + tx;

            if (global_row < M && global_k < K) {
                const std::int64_t a_idx = transpose_a ? (a_base + global_k * lda + global_row) : (a_base + global_row * lda + global_k);
                tile_a[tile_row][tx] = A[a_idx];
            } else {
                tile_a[tile_row][tx] = storage_zero<TA>();
            }
        }

        // Cooperatively load B.
        for (int q = 0; q < 4; ++q) {
            const int tile_k = ty + q * 8;

            const std::int64_t global_k = base_k + tile_k;
            if (global_k < K && col < N) {
                const std::int64_t b_idx = transpose_b ? (b_base + col * ldb + global_k) : (b_base + global_k * ldb + col);
                tile_b[tile_k][tx] = B[b_idx];
            } else {
                tile_b[tile_k][tx] = storage_zero<TB>();
            }
        }
        __syncthreads();

        // Reuse each B value across four output accumulators.
        for (int k = 0; k < BK; ++k) {
            const AccT b = to_accum(tile_b[k][tx]);

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int tile_row = ty + r * 8;
                accum[r] += to_accum(tile_a[tile_row][k]) * b;
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < 4; ++r) {
        const std::int64_t row = row_base + ty + r * 8;

        if (row < M && col < N) {
            const std::int64_t c_idx =c_base + row * ldc + col;

            // const T old_c = (beta == T(0)) ? T(0) : C[c_idx];
            // C[c_idx] = alpha * accum[r] + beta * old_c;
            const AccT scaled_sum =scale_a * scale_b * accum[r];
            AccT result = alpha * scaled_sum;
            if(beta != AccT{0}){
                result += beta * to_accum(C[c_idx]);
            }
            C[c_idx] = from_accum<TC>(result);
        }
    }
}

} // namespace gemm_foundry::detail
