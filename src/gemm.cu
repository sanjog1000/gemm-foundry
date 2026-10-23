#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <cmath>
#include<cuda_fp16.h>
#include <cuda_fp8.h>
#include "gemm_foundry/gemm.hpp"
#include "kernels/gemm_naive.cuh"
#include "kernels/gemm_tiled.cuh"
#include "kernels/gemm_register_blocked.cuh"

namespace gemm_foundry::detail{
namespace{
    /*
    This is the actual multiplication --> calculate actual output elements . 
    This executes on GPU.
    */

    bool is_fp8_data_type(GemmDataType type){
        return type == GemmDataType::fp8_e4m3 || type == GemmDataType::fp8_e5m2 ;
    }

    // Runtime type-combination validation
    bool is_supported_runtime_combination(GemmDataType type_a , GemmDataType type_b , GemmDataType type_c){
        if(type_a == GemmDataType::fp32 && type_b == GemmDataType::fp32 && type_c == GemmDataType::fp32){
            return true;
        }
        if(type_a == GemmDataType::fp64 && type_b == GemmDataType::fp64 && type_c == GemmDataType::fp64){
            return true;
        }
        if(type_a == GemmDataType::fp16 && type_b == GemmDataType::fp16 && (type_c == GemmDataType::fp32 || type_c == GemmDataType::fp16)){
            return true;
        }
        if(is_fp8_data_type(type_a) && is_fp8_data_type(type_b) && (type_c == GemmDataType::fp32 || type_c ==GemmDataType::fp16)){
            return true;
        }
        return false;
    }

    // Host-side validation helpers : calculate the address span
    bool compute_matrix_span(std::int64_t rows , std::int64_t cols , std::int64_t ld , std::int64_t &span){
        if(rows <= 0 || cols <= 0 || ld < cols) return false;

        constexpr auto MAX = std::numeric_limits<std::int64_t>::max();
        if(rows - 1 > (MAX - cols) / ld)return false;

        span = (rows -1) * ld + cols;
        return true;
    }
    //  Validate batch strides
    bool valid_batch_strides(std::int64_t batch_count , std::int64_t stride , std::int64_t span){
        if(stride < 0) return false;
        if(batch_count == 1 && stride == 0) return true;
        if(stride < span) return false;

        constexpr auto MAX = std::numeric_limits<std::int64_t>::max();
        if(batch_count > 1 && batch_count - 1 > (MAX - span) / stride) return false;

        return true;
    }
    
    /*
    Validate input scales :
    a) FP8: finite, strictly positive dequantization scales
    b) Non-FP8: scales must remain 1.0
    */
    bool valid_scale(float scale , GemmDataType type){
        if(is_fp8_data_type(type)){
            return std::isfinite(scale) && scale > 0.0f ;
        }
        return scale == 1.0f ;
    }
/*
This tells the GPU to perform that multiplication.
Its job it to prepare and launch GPU but doesn't calculate the matrix product itself.
*/

Status validate_request(const GemmOptions& options , const void* d_A , const void* d_B ,void* d_C , GemmDataType type_a , GemmDataType type_b , GemmDataType type_c , const GemmScaleOptions& scaling , dim3& grid){
    const auto M = options.M;
    const auto N = options.N;
    const auto K = options.K;

    const auto lda = options.lda;
    const auto ldb = options.ldb;
    const auto ldc = options.ldc;

    const auto transpose_a = options.transpose_a;
    const auto transpose_b = options.transpose_b;

    const auto stride_a = options.stride_a;
    const auto stride_b = options.stride_b;
    const auto stride_c = options.stride_c;

    const auto batch_count = options.batch_count;

    if(M <= 0 || N <= 0 || K <= 0 || batch_count <= 0) return Status::invalid_dimensions;
    if(d_A == nullptr || d_B == nullptr || d_C == nullptr) return Status::null_pointer;
    
    if(!is_supported_runtime_combination(type_a, type_b, type_c)) return Status::unsupported_type;
    if (!valid_scale(scaling.a_dequant_scale, type_a) || !valid_scale(scaling.b_dequant_scale, type_b)) {
        return Status::invalid_scale;
    }
    const std::int64_t a_rows = transpose_a ? K : M;
    const std::int64_t a_cols = transpose_a ? M : K;
    const std::int64_t b_rows = transpose_b ? N : K;
    const std::int64_t b_cols = transpose_b ? K : N;

    std::int64_t span_a = 0;
    std::int64_t span_b = 0;
    std::int64_t span_c = 0;
    
    if (!compute_matrix_span(a_rows, a_cols, lda, span_a) ||
        !compute_matrix_span(b_rows, b_cols, ldb, span_b) ||
        !compute_matrix_span(M, N, ldc, span_c)) {
        return Status::invalid_dimensions;
    }

    if (!valid_batch_strides(batch_count, stride_a, span_a) ||
        !valid_batch_strides(batch_count, stride_b, span_b) ||
        !valid_batch_strides(batch_count, stride_c, span_c)) {
        return Status::invalid_dimensions;
    }

    std::int64_t tile = 16;

    switch (options.algorithm) {
        case Algorithm::naive:
        case Algorithm::tiled:
            tile = 16;
            break;

        case Algorithm::register_blocked:
            tile = 32;
            break;

        default:
            return Status::invalid_algorithm;
    }

    // Standard CUDA grid limits for the grid dimensions used here.
    constexpr std::int64_t MAX_GRID_X = 2147483647LL;
    constexpr std::int64_t MAX_GRID_YZ = 65535LL;

    const std::int64_t grid_x = (N - 1) / tile + 1;
    const std::int64_t grid_y = (M - 1) / tile + 1;

    if (grid_x > MAX_GRID_X || grid_y > MAX_GRID_YZ || batch_count > MAX_GRID_YZ) {
        return Status::invalid_dimensions;
    }
    grid = dim3(static_cast<unsigned int>(grid_x),static_cast<unsigned int>(grid_y),static_cast<unsigned int>(batch_count));

    return Status::success;

}

// information shared by all typed dispatch paths.
struct DispatchParams{
    const GemmOptions& options;
    const void* d_A;
    const void* d_B;
    void* d_C;

    double alpha;
    double beta;

    const GemmScaleOptions& scaling;
    dim3& grid;

};

// Launch one concrete type combination : 
// TA/TB: input storage types.
// TC: output storage type.
// AccT:arithmetic and accumulation type.
template<typename TA , typename TB , typename TC , typename AccT>
Status launch_typed(const DispatchParams& p){
    const TA* A = static_cast<const TA*>(p.d_A);
    const TB* B = static_cast<const TB*>(p.d_B);
    TC* C = static_cast<TC*>(p.d_C);

    const auto& o = p.options;

    const AccT alpha = static_cast<AccT>(p.alpha);
    const AccT beta =static_cast<AccT>(p.beta);
    const AccT scale_a =static_cast<AccT>(p.scaling.a_dequant_scale);
    const AccT scale_b =static_cast<AccT>(p.scaling.b_dequant_scale);

    switch (o.algorithm) {
        case Algorithm::naive:{
            const dim3 block(16, 16);

            gemm_naive_kernel<TA, TB, TC, AccT>
            <<<p.grid, block>>>
            (
                A, B,C,
                o.M, o.N, o.K,
                alpha,beta, scale_a,  scale_b,
                o.lda, o.ldb, o.ldc,
                o.transpose_a,o.transpose_b,
                o.stride_a, o.stride_b,o.stride_c
            );
            break;
        }

        case Algorithm::tiled: {
            constexpr unsigned int TILE = 16;
            const dim3 block(TILE, TILE);

            gemm_tiled_kernel<TA, TB, TC, AccT>
            <<<p.grid, block>>>
            (
                A,B, C,
                o.M,  o.N,  o.K,
                alpha, beta,scale_a, scale_b,
                o.lda,o.ldb, o.ldc,
                o.transpose_a, o.transpose_b,
                o.stride_a, o.stride_b, o.stride_c
            );

            break;
        }

        case Algorithm::register_blocked: {
            const dim3 block(32, 8);

            gemm_register_blocked_kernel<TA, TB, TC, AccT>
            <<<p.grid, block>>>
            (
                A, B, C,
                o.M, o.N, o.K,
                alpha, beta, scale_a, scale_b,
                o.lda, o.ldb, o.ldc,
                o.transpose_a, o.transpose_b,
                o.stride_a, o.stride_b, o.stride_c
            );

            break;
        }
        default:
            return Status::invalid_algorithm;
    }
    const cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        return Status::cuda_error;
    }
    return Status::success;
}

// Dispatch FP16/FP8 inputs to their allowed output storage types. Both paths accumulate in FP32.
template <typename TA , typename TB>
Status dispatch_float_accum_output(const DispatchParams& p , GemmDataType type_c){
    if(type_c == GemmDataType::fp16){
        return launch_typed<TA ,TB, __half, float>(p);
    }
    if(type_c == GemmDataType::fp32){
        return launch_typed<TA ,TB, float, float>(p);
    }
    return Status::unsupported_type;
}

// Dispatch validated type combinations to concrete template instantiations
Status dispatch_types(const DispatchParams& p , GemmDataType type_a , GemmDataType type_b , GemmDataType type_c){
    // FP32 x FP32 -> FP32
    if(type_a == GemmDataType::fp32 && type_b == GemmDataType::fp32 && type_c == GemmDataType::fp32){
        return launch_typed<float , float , float , float>(p);
    }
    // FP64 x FP64 -> FP64
    if(type_a == GemmDataType::fp64 && type_b == GemmDataType::fp64 && type_c == GemmDataType::fp64){
        return launch_typed<double , double , double , double>(p);
    }
    // FP16 x FP16 -> FP16 or FP32
    if(type_a == GemmDataType::fp16 && type_b == GemmDataType::fp16 ){
        return dispatch_float_accum_output<__half , __half>(p , type_c);
    }
    // E4M3 x E4M3.
    if (type_a == GemmDataType::fp8_e4m3 && type_b == GemmDataType::fp8_e4m3) {
        return dispatch_float_accum_output< __nv_fp8_e4m3, __nv_fp8_e4m3>(p, type_c);
    }
    // E5M2 x E5M2.
    if (type_a == GemmDataType::fp8_e5m2 && type_b == GemmDataType::fp8_e5m2) {
        return dispatch_float_accum_output< __nv_fp8_e5m2, __nv_fp8_e5m2>(p, type_c);
    }
    // E4M3 x E5M2.
    if (type_a == GemmDataType::fp8_e4m3 && type_b == GemmDataType::fp8_e5m2) {
        return dispatch_float_accum_output<__nv_fp8_e4m3, __nv_fp8_e5m2>(p, type_c);
    }
    // E5M2 x E4M3.
    if (type_a == GemmDataType::fp8_e5m2 && type_b == GemmDataType::fp8_e4m3) {
        return dispatch_float_accum_output<__nv_fp8_e5m2, __nv_fp8_e4m3>(p, type_c);
    }
    return Status::unsupported_type;
}

}   // anonymous namespace

Status gemm_dispatch_impl(const GemmOptions& options , const void* d_A ,const void* d_B , void* d_C , double alpha , double beta , GemmDataType type_a , GemmDataType type_b , GemmDataType type_c, const GemmScaleOptions& scaling){
    if(!is_supported_runtime_combination(type_a,type_b,type_c)){
        return Status::unsupported_type;
    }
    dim3 grid{};

    const Status validation = validate_request(options,d_A,d_B,d_C,type_a,type_b,type_c, scaling ,grid);

    if(validation != Status::success){
        return validation;
    }

    const DispatchParams params{options,d_A,d_B,d_C , alpha , beta , scaling , grid};
    return dispatch_types(params, type_a,type_b ,type_c);
}

}// namespace gemm_foundry::detail