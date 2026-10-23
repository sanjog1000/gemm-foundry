
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gemm_foundry/gemm.hpp"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        const cudaError_t error = (call);                                  \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__   \
                      << ": " << cudaGetErrorString(error) << '\n';         \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    } while (0)

const char* algorithm_name(gemm_foundry::Algorithm algorithm) {
    using gemm_foundry::Algorithm;

    switch (algorithm) {
        case Algorithm::naive:
            return "naive";
        case Algorithm::tiled:
            return "tiled";
        case Algorithm::register_blocked:
            return "register_blocked";
        default:
            return "unknown";
    }
}

int main() {
    using namespace gemm_foundry;

    // Compute C = alpha * A * B + beta * C.
    constexpr std::int64_t M = 2;
    constexpr std::int64_t N = 2;
    constexpr std::int64_t K = 3;

    constexpr float alpha = 1.25f;
    constexpr float beta  = 0.5f;
    constexpr float tolerance = 1.0e-4f;

    // Row-major A: M x K; B: K x N; C: M x N.
    const std::vector<float> A = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    const std::vector<float> B = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f
    };

    const std::vector<float> C_initial = {
        0.5f, -1.0f,
        2.0f,  0.25f
    };

    // CPU reference, including alpha and beta.
    std::vector<float> expected(M * N, 0.0f);

    for (std::int64_t row = 0; row < M; ++row) {
        for (std::int64_t col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (std::int64_t k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            const auto index = row * N + col;
            expected[index] =
                alpha * sum + beta * C_initial[index];
        }
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_A), A.size() * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_B), B.size() * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_C), C_initial.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

    const std::array<Algorithm, 3> algorithms = {
        Algorithm::naive,
        Algorithm::tiled,
        Algorithm::register_blocked
    };

    bool all_passed = true;

    for (const Algorithm algorithm : algorithms) {
        // Reset C because beta uses its initial contents.
        CUDA_CHECK(cudaMemcpy(
            d_C,
            C_initial.data(),
            C_initial.size() * sizeof(float),
            cudaMemcpyHostToDevice));

        GemmOptions options;
        options.M = M;
        options.N = N;
        options.K = K;

        options.lda = K;
        options.ldb = N;
        options.ldc = N;

        options.algorithm = algorithm;

        const Status status = gemm<float, float, float>(
            options, d_A, d_B, d_C, alpha, beta);

        if (status != Status::success) {
            std::cerr << algorithm_name(algorithm)
                      << ": gemm() failed with status "
                      << static_cast<int>(status) << '\n';
            all_passed = false;
            continue;
        }

        // gemm() launches asynchronously; synchronize to catch execution errors.
        const cudaError_t sync_error = cudaDeviceSynchronize();

        if (sync_error != cudaSuccess) {
            std::cerr << algorithm_name(algorithm)
                      << ": kernel execution failed: "
                      << cudaGetErrorString(sync_error) << '\n';
            all_passed = false;
            continue;
        }

        std::vector<float> actual(M * N);

        CUDA_CHECK(cudaMemcpy(
            actual.data(),
            d_C,
            actual.size() * sizeof(float),
            cudaMemcpyDeviceToHost));

        bool passed = true;

        for (std::size_t i = 0; i < actual.size(); ++i) {
            const float error = std::abs(actual[i] - expected[i]);
            const float limit =
                tolerance * std::max(1.0f, std::abs(expected[i]));

            std::cout << algorithm_name(algorithm)
                      << " C[" << i << "] = " << actual[i]
                      << ", expected = " << expected[i]
                      << ", error = " << error << '\n';

            if (!std::isfinite(actual[i]) || error > limit) {
                passed = false;
            }
        }

        std::cout << algorithm_name(algorithm) << ": "
                  << (passed ? "PASS" : "FAIL") << "\n\n";

        all_passed = all_passed && passed;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << (all_passed
        ? "ALL GEMM TESTS PASSED\n"
        : "ONE OR MORE GEMM TESTS FAILED\n");

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
