#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "gemm_foundry/gemm.hpp"

using namespace gemm_foundry;

static bool cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": "
                  << cudaGetErrorString(error) << '\n';
        return false;
    }
    return true;
}

static const char* algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::naive: return "naive";
        case Algorithm::tiled: return "tiled";
        case Algorithm::register_blocked: return "register_blocked";
        default: return "unknown";
    }
}

static float a_value(int batch, int i, int k) {
    return static_cast<float>(
        ((batch + 1) * 7 + i * 3 + k * 5) % 17 - 8) * 0.125f;
}

static float b_value(int batch, int k, int j) {
    return static_cast<float>(
        ((batch + 2) * 11 + k * 2 + j * 7) % 19 - 9) * 0.125f;
}

static float c_value(int batch, int i, int j) {
    return static_cast<float>(
        ((batch + 3) * 5 + i * 7 + j * 3) % 13 - 6) * 0.25f;
}

static bool run_case(
    const std::string& name,
    Algorithm algorithm,
    int M, int N, int K,
    bool transpose_a,
    bool transpose_b,
    int batch_count,
    float alpha,
    float beta)
{
    const int64_t a_rows = transpose_a ? K : M;
    const int64_t a_cols = transpose_a ? M : K;
    const int64_t b_rows = transpose_b ? N : K;
    const int64_t b_cols = transpose_b ? K : N;

    const int64_t lda = a_cols;
    const int64_t ldb = b_cols;
    const int64_t ldc = N;

    const int64_t a_size = a_rows * lda;
    const int64_t b_size = b_rows * ldb;
    const int64_t c_size = static_cast<int64_t>(M) * ldc;

    std::vector<float> h_A(batch_count * a_size, 0.0f);
    std::vector<float> h_B(batch_count * b_size, 0.0f);
    std::vector<float> h_C(batch_count * c_size, 0.0f);
    std::vector<float> expected(batch_count * c_size, 0.0f);

    // Build physical A and B storage according to the transpose flags.
    for (int batch = 0; batch < batch_count; ++batch) {
        const int64_t a_base = batch * a_size;
        const int64_t b_base = batch * b_size;
        const int64_t c_base = batch * c_size;

        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                const int64_t index = transpose_a
                    ? a_base + k * lda + i
                    : a_base + i * lda + k;

                h_A[index] = a_value(batch, i, k);
            }
        }

        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                const int64_t index = transpose_b
                    ? b_base + j * ldb + k
                    : b_base + k * ldb + j;

                h_B[index] = b_value(batch, k, j);
            }
        }

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                h_C[c_base + i * ldc + j] = c_value(batch, i, j);
            }
        }
    }

    // Independent CPU reference: C = alpha * op(A) * op(B) + beta * C.
    expected = h_C;

    for (int batch = 0; batch < batch_count; ++batch) {
        const int64_t a_base = batch * a_size;
        const int64_t b_base = batch * b_size;
        const int64_t c_base = batch * c_size;

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;

                for (int k = 0; k < K; ++k) {
                    const int64_t ai = transpose_a
                        ? a_base + k * lda + i
                        : a_base + i * lda + k;

                    const int64_t bi = transpose_b
                        ? b_base + j * ldb + k
                        : b_base + k * ldb + j;

                    sum += h_A[ai] * h_B[bi];
                }

                const int64_t ci = c_base + i * ldc + j;
                expected[ci] = alpha * sum + beta * h_C[ci];
            }
        }
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    auto cleanup = [&]() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
    };

    if (!cuda_check(cudaMalloc(&d_A, h_A.size() * sizeof(float)),
                    "allocate A") ||
        !cuda_check(cudaMalloc(&d_B, h_B.size() * sizeof(float)),
                    "allocate B") ||
        !cuda_check(cudaMalloc(&d_C, h_C.size() * sizeof(float)),
                    "allocate C")) {
        cleanup();
        return false;
    }

    if (!cuda_check(cudaMemcpy(d_A, h_A.data(),
                              h_A.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy A") ||
        !cuda_check(cudaMemcpy(d_B, h_B.data(),
                              h_B.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy B") ||
        !cuda_check(cudaMemcpy(d_C, h_C.data(),
                              h_C.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "copy C")) {
        cleanup();
        return false;
    }

    GemmOptions options;
    options.M = M;
    options.N = N;
    options.K = K;
    options.lda = lda;
    options.ldb = ldb;
    options.ldc = ldc;
    options.transpose_a = transpose_a;
    options.transpose_b = transpose_b;
    options.algorithm = algorithm;
    options.batch_count = batch_count;

    // Strides are measured in elements, not bytes.
    if (batch_count > 1) {
        options.stride_a = a_size;
        options.stride_b = b_size;
        options.stride_c = c_size;
    }

    const Status status = gemm<float, float, float>(
        options, d_A, d_B, d_C, alpha, beta);

    if (status != Status::success) {
        std::cerr << "FAIL | " << name << " | "
                  << algorithm_name(algorithm)
                  << " | GEMM returned status "
                  << static_cast<int>(status) << '\n';
        cleanup();
        return false;
    }

    if (!cuda_check(cudaDeviceSynchronize(), "kernel execution")) {
        cleanup();
        return false;
    }

    std::vector<float> actual(h_C.size());

    if (!cuda_check(cudaMemcpy(actual.data(), d_C,
                              actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy result")) {
        cleanup();
        return false;
    }

    cleanup();

    float max_error = 0.0f;

    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);

        const float tolerance =
            1.0e-4f + 1.0e-4f * std::abs(expected[i]);

        if (!std::isfinite(actual[i]) || error > tolerance) {
            std::cerr << "FAIL | " << name << " | "
                      << algorithm_name(algorithm)
                      << " | index=" << i
                      << " actual=" << actual[i]
                      << " expected=" << expected[i]
                      << " error=" << error << '\n';
            return false;
        }
    }

    std::cout << "PASS | " << name << " | "
              << algorithm_name(algorithm)
              << " | max_abs_error=" << max_error << '\n';

    return true;
}

int main() {
    const Algorithm algorithms[] = {
        Algorithm::naive,
        Algorithm::tiled,
        Algorithm::register_blocked
    };

    bool all_passed = true;

    for (Algorithm algorithm : algorithms) {
        all_passed &= run_case(
            "irregular_dimensions", algorithm,
            19, 23, 17, false, false, 1, 1.25f, 0.5f);

        all_passed &= run_case(
            "transpose_a", algorithm,
            7, 11, 5, true, false, 1, 0.75f, 0.25f);

        all_passed &= run_case(
            "transpose_b_beta_zero", algorithm,
            7, 11, 5, false, true, 1, 1.5f, 0.0f);

        all_passed &= run_case(
            "transpose_both_beta_one", algorithm,
            7, 11, 5, true, true, 1, 1.0f, 1.0f);

        all_passed &= run_case(
            "batched_gemm", algorithm,
            5, 7, 3, false, false, 3, 1.25f, 0.5f);
    }

    if (!all_passed) {
        std::cerr << "\nGEMM TESTS FAILED\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nALL GEMM TESTS PASSED\n";
    return EXIT_SUCCESS;
}