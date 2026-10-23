#include <gemm_foundry/gemm.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t error = (call);                                \
        if (error != cudaSuccess) {                                \
            std::cerr << "CUDA error: "                            \
                      << cudaGetErrorString(error) << '\n';        \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    } while (0)

// --------------------------------------------------
// Check the status returned by our GEMM library.
// --------------------------------------------------

void check_status(gemm_foundry::Status status) {
    if (status != gemm_foundry::Status::success) {
        std::cerr << "GEMM failed. Status = "
                  << static_cast<int>(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// --------------------------------------------------
// Generate random FP32 matrix elements.
// --------------------------------------------------

void fill_random(
    std::vector<float>& values,
    std::mt19937& rng
) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& value : values) {
        value = dist(rng);
    }
}

// --------------------------------------------------
// CPU reference implementation.
//
// C = A * B
//
// A: M x K
// B: K x N
// C: M x N
//
// All matrices use row-major storage.
// --------------------------------------------------

void cpu_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    std::int64_t M,
    std::int64_t N,
    std::int64_t K
) {
    for (std::int64_t row = 0; row < M; ++row) {
        for (std::int64_t col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (std::int64_t k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

// --------------------------------------------------
// Construct options for the current GEMM operation.
// --------------------------------------------------

gemm_foundry::GemmOptions make_options(
    std::int64_t M,
    std::int64_t N,
    std::int64_t K,
    gemm_foundry::Algorithm algorithm
) {
    gemm_foundry::GemmOptions options{};

    options.M = M;
    options.N = N;
    options.K = K;

    // Row-major matrices without padding.
    options.lda = K;
    options.ldb = N;
    options.ldc = N;

    options.algorithm = algorithm;

    options.transpose_a = false;
    options.transpose_b = false;

    // One matrix multiplication.
    options.batch_count = 1;

    // Zero strides are permitted for a single batch.
    options.stride_a = 0;
    options.stride_b = 0;
    options.stride_c = 0;

    return options;
}

// --------------------------------------------------
// Launch the selected GEMM implementation.
// --------------------------------------------------

void run_gemm(
    const gemm_foundry::GemmOptions& options,
    const float* d_A,
    const float* d_B,
    float* d_C
) {
    check_status(
        gemm_foundry::gemm<float>(
            options,
            d_A,
            d_B,
            d_C,
            1.0f,  // alpha
            0.0f   // beta
        )
    );
}

// --------------------------------------------------
// Benchmark one selected GPU kernel.
// --------------------------------------------------

void benchmark_algorithm(
    const char* name,
    gemm_foundry::Algorithm algorithm,
    std::int64_t M,
    std::int64_t N,
    std::int64_t K,
    const float* d_A,
    const float* d_B,
    float* d_C,
    const std::vector<float>& expected,
    std::vector<float>& output,
    int repetitions
) {
    const auto options = make_options(M, N, K, algorithm);

    // ----------------------------------------------
    // First launch: correctness verification.
    // ----------------------------------------------

    run_gemm(options, d_A, d_B, d_C);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(
            output.data(),
            d_C,
            output.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    float max_error = 0.0f;

    for (std::size_t i = 0; i < output.size(); ++i) {
        max_error = std::max(
            max_error,
            std::fabs(output[i] - expected[i])
        );
    }

    constexpr float TOLERANCE = 1e-3f;

    if (max_error > TOLERANCE) {
        std::cerr << name << " correctness check failed.\n"
                  << "Maximum absolute error: "
                  << std::scientific << max_error << '\n';

        std::exit(EXIT_FAILURE);
    }

    // ----------------------------------------------
    // Second phase: measure kernel execution time.
    // ----------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repetitions; ++i) {
        run_gemm(options, d_A, d_B, d_C);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(&total_ms, start, stop)
    );

    const double average_ms =
        static_cast<double>(total_ms) / repetitions;

    const double operations =
        2.0
        * static_cast<double>(M)
        * static_cast<double>(N)
        * static_cast<double>(K);

    const double gflops =
        operations / (average_ms * 1.0e6);

    std::cout << std::left << std::setw(10) << name
              << " | "
              << std::right << std::fixed
              << std::setprecision(6)
              << std::setw(12) << average_ms
              << " ms | "
              << std::setprecision(2)
              << std::setw(9) << gflops
              << " GFLOP/s | max error: "
              << std::scientific << max_error
              << " | PASS\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// --------------------------------------------------
// Allocate matrices, compute the CPU reference,
// and benchmark all three algorithms.
// --------------------------------------------------

void benchmark_case(
    std::int64_t M,
    std::int64_t N,
    std::int64_t K,
    int repetitions,
    std::mt19937& rng
) {
    std::vector<float> A(
        static_cast<std::size_t>(M * K)
    );

    std::vector<float> B(
        static_cast<std::size_t>(K * N)
    );

    std::vector<float> expected(
        static_cast<std::size_t>(M * N),
        0.0f
    );

    std::vector<float> output(
        static_cast<std::size_t>(M * N),
        0.0f
    );

    fill_random(A, rng);
    fill_random(B, rng);

    // Calculate the expected result on the CPU.
    cpu_gemm(A, B, expected, M, N, K);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&d_A),
            A.size() * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&d_B),
            B.size() * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&d_C),
            output.size() * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            A.data(),
            A.size() * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            B.data(),
            B.size() * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    std::cout << "\nMatrix: "
              << M << " x " << N << " x " << K << '\n';

    std::cout << "Algorithm  | Average time | Throughput\n";

    benchmark_algorithm(
        "Naive",
        gemm_foundry::Algorithm::naive,
        M, N, K,
        d_A, d_B, d_C,
        expected, output, repetitions
    );

    benchmark_algorithm(
        "Tiled",
        gemm_foundry::Algorithm::tiled,
        M, N, K,
        d_A, d_B, d_C,
        expected, output, repetitions
    );

    benchmark_algorithm(
        "RegBlock",
        gemm_foundry::Algorithm::register_blocked,
        M, N, K,
        d_A, d_B, d_C,
        expected, output, repetitions
    );

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// --------------------------------------------------
// Entry point.
// --------------------------------------------------

int main() {
    int device = 0;

    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};

    CUDA_CHECK(
        cudaGetDeviceProperties(&properties, device)
    );

    std::cout << "GPU: " << properties.name << '\n';

    std::cout
        << "Comparing naive, tiled, and register-blocked GEMM\n";

    std::mt19937 rng(2026);

    benchmark_case(64, 64, 64, 100, rng);
    benchmark_case(128, 128, 128, 100, rng);
    benchmark_case(256, 256, 256, 30, rng);
    benchmark_case(512, 512, 512, 10, rng);

    return EXIT_SUCCESS;
}