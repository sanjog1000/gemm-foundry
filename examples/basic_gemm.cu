#include<cuda_runtime.h>
#include <cmath>
#include<iostream>
#include<vector>
#include<cstdio>
#include<cstdlib>
#include <algorithm>
#include <iomanip>
#include <random>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t error = (call);                                   \
        if (error != cudaSuccess) {                                   \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                         __FILE__, __LINE__,                           \
                         cudaGetErrorString(error));                  \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

__global__ void gemm_naive(const float* A , const float* B , float* C , int M , int N ,int K){
    const int col = blockIdx.x * blockDim.x + threadIdx.x ;
    const int row = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row >= M || col >= N)return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float a = A[row * K + k];
        const float b = B[k * N + col];
        sum += a * b;
    }
    C[row * N + col] = sum;
}
void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,std::vector<float>& C,int M,int N,int K){
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}
void fill_random(std::vector<float>& values ,std::mt19937& rng){
    std::uniform_real_distribution<float>dist(-1.0f,1.0f);
    for(float& value : values){
        value = dist(rng);
    }
}
void corectness_test(int M ,int K , int N){
    std::mt19937 rng(2026);

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> expected(M * N, 0.0f);

    fill_random(A , rng);
    fill_random(A , rng);
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cpu_gemm(A, B, expected, M, N, K);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A),A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B),B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C),C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),cudaMemcpyHostToDevice));

    const dim3 block(16,16);
    const dim3 grid(
        (N + block.x - 1) / block.x ,
        (N + block.y - 1) / block.y
    );
}
int main(){
    const int M = 2;
    const int N = 2;
    const int K = 3;

    std::vector<float> A={
        1, 2, 3,
        4, 5, 6
    };
    std::vector<float> B={
        7, 8,
        9, 10,
        11, 12
    };

    std::vector<float> C(M * N, 0.0f);
    std::vector<float> expected(M * N, 0.0f);

    cpu_gemm(A, B, expected, M, N, K);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A),A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B),B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C),C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),cudaMemcpyDeviceToHost));
    bool passed = true;

    for (int i = 0; i < M * N; ++i) {
        std::cout << "C[" << i << "] = " << C[i] << ", expected = " << expected[i] << '\n';
        if (std::fabs(C[i] - expected[i]) > 1e-5f) {
            passed = false;
        }
    }
    std::cout << (passed ? "TEST PASSED\n" : "TEST FAILED\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}