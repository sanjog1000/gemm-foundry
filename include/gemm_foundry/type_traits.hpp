#pragma once

#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "gemm_foundry/types.hpp"

namespace gemm_foundry::detail {
template<typename T>
struct GemmTypeInfo{
    static constexpr bool supported = false;
    static constexpr GemmDataType value = GemmDataType::unsupported;
};

template<>
struct GemmTypeInfo<float>{
    static constexpr bool supported = true;
    static constexpr GemmDataType value = GemmDataType::fp32;
};

template<>
struct GemmTypeInfo<double>{
    static constexpr bool supported = true;
    static constexpr GemmDataType value = GemmDataType::fp64;
};

template<>
struct GemmTypeInfo<__half>{
    static constexpr bool supported = true;
    static constexpr GemmDataType value = GemmDataType::fp16;
};

template<>
struct GemmTypeInfo<__nv_fp8_e5m2>{
    static constexpr bool supported = true;
    static constexpr GemmDataType value = GemmDataType::fp8_e5m2;
};

template<>
struct GemmTypeInfo<__nv_fp8_e4m3>{
    static constexpr bool supported = true;
    static constexpr GemmDataType value = GemmDataType::fp8_e4m3;
};

template <typename T>
inline constexpr bool is_fp8_type_v = std::is_same_v<T , __nv_fp8_e4m3> || std::is_same_v<T , __nv_fp8_e5m2> ;

// Select the compute/accumulation type : 
// FP64 x FP64 -> FP64
// All other currently intended input combinations -> FP32
// If both TA and TB are double then only double , otherwise for all other cases float
template <typename TA, typename TB>
using ComputeTypeT = std::conditional_t<std::is_same_v<TA , double> && std::is_same_v<TB , double> , double , float>;

// Decide which combinations the public API permits : 
//   float x float       -> float output
//   double x double     -> double output
//   FP16 x FP16         -> FP16 or FP32 output
//   FP8 x FP8           -> FP16 or FP32 output
template <typename TA , typename TB , typename TC>
inline constexpr bool is_supported_gemm_types_v = 
    (
        std::is_same_v<TA , float> &&
        std::is_same_v<TB , float> &&
        std::is_same_v<TC , float> 
    ) ||
    (
        std::is_same_v<TA , double> &&
        std::is_same_v<TB , double> &&
        std::is_same_v<TC , double> 
    ) ||
    (
        std::is_same_v<TA , __half> &&
        std::is_same_v<TB , __half> &&
        (
            std::is_same_v<TC , __half> ||
            std::is_same_v<TC , float> 
        )
    ) ||
    (
        is_fp8_type_v<TA> &&
        is_fp8_type_v<TB> &&
        (
            std::is_same_v<TC , __half> ||
            std::is_same_v<TC , float>
        )
    );
}  // namespace gemm_foundry::detail