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
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

#define FLT_MIN 1.175494351e-38F
#define FLT_MAX 3.402823466e+38F
#define FLT_EPSILON 1.192092896e-07F

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

void softmax(hipStream_t stream, const argument& result, const argument& arg, int64_t axis)
{
    std::vector<std::size_t> dims = arg.get_shape().lens();
    int count                     = arg.get_shape().elements();
    axis                          = static_cast<int>((axis + dims.size()) % dims.size());
    int channel                   = dims[axis];

    // 计算outer_num和inner_num
    int outer_num = 1;
    for(int index = 0; index < axis; ++index)
    {
        outer_num *= dims[index];
    }
    int inner_num = 1;
    for(int index = axis + 1; index < dims.size(); ++index)
    {
        inner_num *= dims[index];
    }
    int num_threads = 128; // 每个block的线程数

    // 创建临时变量
    const migraphx::argument temp_buffer =
        migraphx::gpu::to_gpu(migraphx::generate_argument(arg.get_shape()));

    // 计算softmax_channel_max_kernel
    hip_visit_all(temp_buffer, arg)([&](auto output, auto input) {
        gs_launch(stream, outer_num * inner_num, num_threads)([=](auto i, auto idx) __device__ {
            int n        = i / inner_num;
            int s        = i % inner_num;
            float maxval = -FLT_MAX;
            for(int c = 0; c < channel; ++c)
            {
                maxval = MAX(input[(n * channel + c) * inner_num + s], maxval);
            }
            output[i] = maxval;
        }

        );
    });

    // 计算softmax_channel_subtract_exp_kernel
    hip_visit_all(result, arg, temp_buffer)([&](auto output, auto input, auto channel_max) {
        gs_launch(stream, count, num_threads)([=](auto i, auto idx) __device__ {
            int n     = i / channel / inner_num;
            int s     = i % inner_num;
            output[i] = exp(input[i] - channel_max[n * inner_num + s]);
        });
    });

    // 计算softmax_channel_sum_kernel
    hip_visit_all(result, arg, temp_buffer)([&](auto output, auto input, auto channel_sum) {
        gs_launch(stream, outer_num * inner_num, num_threads)([=](auto i, auto idx) __device__ {
            int n     = i / inner_num;
            int s     = i % inner_num;
            float sum = 0;
            for(int c = 0; c < channel; ++c)
            {
                sum += output[(n * channel + c) * inner_num + s];
            }
            channel_sum[i] = sum;
        });
    });

    // 计算softmax_channel_div_kernel
    hip_visit_all(result, arg, temp_buffer)([&](auto output, auto input, auto channel_sum) {
        gs_launch(stream, count, num_threads)([=](auto i, auto idx) __device__ {
            int n = i / channel / inner_num;
            int s = i % inner_num;
            output[i] /= channel_sum[n * inner_num + s];
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
