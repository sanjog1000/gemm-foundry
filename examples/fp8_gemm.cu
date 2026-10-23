#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "gemm_foundry/gemm.hpp"

void run_example(
    const __nv_fp8_e4m3* d_A,
    const __nv_fp8_e5m2* d_B,
    __half* d_C
) {
    gemm_foundry::GemmOptions options{};

    options.M = 128;
    options.N = 128;
    options.K = 128;

    options.lda = 128;
    options.ldb = 128;
    options.ldc = 128;

    options.algorithm =
        gemm_foundry::Algorithm::tiled;

    options.batch_count = 1;

    gemm_foundry::GemmScaleOptions scaling{};

    scaling.a_dequant_scale = 0.01f;
    scaling.b_dequant_scale = 0.02f;

    const gemm_foundry::Status status =
        gemm_foundry::gemm(
            options,
            d_A,
            d_B,
            d_C,
            1.0f,   // alpha is FP32
            0.0f,   // beta is FP32
            scaling
        );

    // Check status in the calling code.
    (void)status;
}