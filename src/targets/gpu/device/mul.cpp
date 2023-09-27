/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/device/mul.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void mul(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ { return x * y; });
}

void mul(hipStream_t stream,
         const argument& result,
         const argument& arg1,
         const argument& arg2,
         const argument& arg3)
{
    nary(stream, result, arg1, arg2, arg3)([](auto x, auto y, auto z)
                                               __device__ { return x * y * z; });
}

void pow_mul_add_mul_tanh_add_mul_mul(hipStream_t stream,
             const argument& result,
             const argument& arg1,
             const argument& arg2,
             const argument& arg3,
             const argument& arg4,
             const argument& arg5,
             const argument& arg6,
             const argument& arg7,
             const argument& arg8)
{
    nary(stream, result, arg1, arg2,arg3,arg4,arg5,arg6,arg7,arg8)([](auto x1, auto x2,auto x3,auto x4,auto x5,auto x6,auto x7,auto x8) __device__ 
    {
        // pow_mul_add_mul_tanh_add
        auto pow1 = ::pow(to_hip_type(x1), to_hip_type(x2)); 
        auto pow_mul_add_mul_tanh_add = ::tanh(to_hip_type(((pow1*x3)+x4)*x5))+x6;
        
        // mul1
        auto mul1=x7*x8;
            
        return mul1*pow_mul_add_mul_tanh_add;
    });

}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
