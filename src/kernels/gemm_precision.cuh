#pragma once

#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace gemm_foundry::detail{
    template<typename T>
    struct AccumulationType{
        using type = T;
    };

    template<>
    struct AccumulationType<__half>{
        using type = float;
    };

    template<>
    struct AccumulationType<__nv_fp8_e5m2>{
        using type = float;
    };

    template<>
    struct AccumulationType<__nv_fp8_e4m3>{
        using type = float;
    };

    template <typename T>
    using AccumulationTypeT = typename AccumulationType<T>::type;

    // Convert storage type to accumulation type(float)
    template<typename T>
    __device__ __forceinline__
    AccumulationTypeT<T> to_accum(T value){
        if constexpr(std::is_same_v<T , __half>){
            return __half2float(value);
        }else if constexpr(std::is_same_v<T , __nv_fp8_e4m3> || std::is_same_v<T , __nv_fp8_e5m2>){
            return static_cast<float>(value);
        }else{
            return static_cast<AccumulationTypeT<T>>(value);
        }
    } 
    
    // Convert the result back to the storage type
    template<typename T>
    __device__ __forceinline__
    T from_accum(AccumulationTypeT<T> value){
        if constexpr(std::is_same_v<T , __half>){
            return __float2half_rn(value);
        }else{
            return static_cast<T>(value);      // assuming output matrix is not FP8
        }
    }

    // Produce zero in a matrix's storage type
    template<typename T>
    __device__ __forceinline__
    T storage_zero(){
        if constexpr(std::is_same_v<T , __half>){
            return __float2half_rn(0.0f);
        }else if constexpr (std::is_same_v<T , __nv_fp8_e4m3>){
            return __nv_fp8_e4m3(0.0f);
        }else if constexpr (std::is_same_v<T , __nv_fp8_e5m2>){
            return __nv_fp8_e5m2(0.0f);
        }else{
            return T{0};
        }
    }
}  // namespace gemm_foundry::detail