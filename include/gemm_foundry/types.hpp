#pragma once
#include<cstdint>

namespace gemm_foundry{
    enum class Status {
        success,
        invalid_dimensions,
        null_pointer,
        cuda_error,
        invalid_algorithm,
        invalid_scale,
        unsupported_type
    };

    enum class Algorithm {
        naive,
        tiled,
        register_blocked
    };

    enum class GemmDataType {
        unsupported,
        fp16,
        fp32,
        fp64,
        fp8_e4m3,
        fp8_e5m2
    };

// Computes C = alpha * op(A) * op(B) + beta * C
// Logical dimensions:
//   op(A): M x K
//   op(B): K x N
//   C:     M x N
// Matrices use row-major storage.
// lda, ldb, ldc are physical row strides in elements.

    struct GemmOptions {
        std::int64_t M =0;
        std::int64_t N = 0;
        std::int64_t K= 0;

        std::int64_t lda = 0;   // physical row stride of A
        std::int64_t ldb = 0;   // physical row stride of B
        std::int64_t ldc = 0;   // physical row stride of C

        Algorithm algorithm = Algorithm::tiled;

        bool transpose_a = false;
        bool transpose_b = false;

        // no. of independent matrix multiplications
        std::int64_t batch_count = 1;

        // dist btw consecutive matrices, in elements
        std::int64_t stride_a = 0;
        std::int64_t stride_b = 0;
        std::int64_t stride_c = 0;
    };

    // A_real ~= a_dequant_scale * A_quantized
    // B_real ~= b_dequant_scale * B_quantized

    struct GemmScaleOptions{
        float a_dequant_scale =1.0f;
        float b_dequant_scale =1.0f;
    };
}  // namespace gemm_foundry
