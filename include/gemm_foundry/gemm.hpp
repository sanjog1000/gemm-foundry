// This is the public interface of our GEMM library
/*
This tells what operation they can request , 
what inputs it needs and results does it report.
*/
#pragma once
#include <cstdint>
#include<type_traits>
#include "gemm_foundry/types.hpp"
#include "gemm_foundry/type_traits.hpp"
#include <cuda_fp16.h>

namespace gemm_foundry::detail{
    // Private compiled implementation

    Status gemm_dispatch_impl(const GemmOptions& options , const void* d_A , const void*d_B , void* d_C , double alpha , double beta , GemmDataType type_a , GemmDataType type_b , GemmDataType type_c, const GemmScaleOptions& scaling);
}  // namespace gemm_foundry::detail


namespace gemm_foundry{
// Computes:
// C = alpha * s_A * s_B * op(A) * op(B) + beta * C
// For non-FP8 inputs, s_A = s_B = 1
// The accumulation type is selected from TA and TB.
// C has its own independent storage type TC.

    template<typename TA, typename TB , typename TC>
    Status gemm(const GemmOptions& options , const TA* d_A , const TB* d_B , TC* d_C, detail::ComputeTypeT<TA , TB>alpha , detail::ComputeTypeT<TA ,TB>beta , const GemmScaleOptions& scaling = GemmScaleOptions{}){
        static_assert(detail::GemmTypeInfo<TA>::supported , "gemm: unsupported storage type for A.");

        static_assert(detail::GemmTypeInfo<TB>::supported , "gemm: unsupported storage type for B.");

        static_assert(detail::GemmTypeInfo<TC>::supported , "gemm: unsupported storage type for C.");

        static_assert(detail::is_supported_gemm_types_v<TA,TB,TC>, "gemm: unsupported combination of input and output types.");

        return detail::gemm_dispatch_impl(
            options,
            static_cast<const void*>(d_A),
            static_cast<const void*>(d_B),
            static_cast<void*>(d_C),
            static_cast<double>(alpha),
            static_cast<double>(beta),
            detail::GemmTypeInfo<TA>::value,
            detail::GemmTypeInfo<TB>::value,
            detail::GemmTypeInfo<TC>::value,
            scaling
        );    
    }
}