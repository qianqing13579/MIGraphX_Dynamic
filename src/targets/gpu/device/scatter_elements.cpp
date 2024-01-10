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
#include <migraphx/gpu/device/scatter_elements.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument scatter_elements(hipStream_t stream,
                          argument result,
                          std::vector<argument> args,
                          int64_t axis,
                          int64_t reduction)
{
    auto input_shape                       = args[0].get_shape();
    std::vector<std::size_t> input_strides = input_shape.strides();

    auto indices_shape = args[1].get_shape();
    auto num_dim       = indices_shape.lens().size();

    int64_t axis_dim_size = input_shape.lens()[axis];
    hip_visit_all(result, args[0], args[2])([&](auto output, auto data, auto update) {
        auto* output_ptr     = device_cast(output.data());
        const auto* data_ptr = device_cast(data.data());
        const auto* upd_ptr  = device_cast(update.data());
        gs_launch(stream,
                  input_shape.elements())([=](auto i) __device__ { output_ptr[i] = data_ptr[i]; });
        hip_visit_all(args[1])([&](auto indices) {
            const auto* indices_ptr = device_cast(indices.data());
            using hip_index         = typename decltype(output)::hip_index;
            hip_index input_stride_1;
            std::copy(input_strides.begin(), input_strides.end(), input_stride_1.begin());

            gs_launch(stream, indices_shape.elements())([=](auto id) __device__ {
                int64_t offset = 0;
                auto out_idx   = indices.get_shape().multi(id);

                // 边界控制
                auto index = indices_ptr[id];
                if(index >= -axis_dim_size && index < axis_dim_size)
                {
                    index = index < 0 ? index + axis_dim_size : index;

                    for(int i = 0; i < num_dim; i++)
                    {
                        if(i == axis)
                        {
                            offset += (index * input_stride_1[i]);
                        }
                        else
                        {
                            offset += (out_idx[i] * input_stride_1[i]);
                        }
                    }

                    switch(reduction)
                    {
                    case 0: // none模式
                        output_ptr[offset] = upd_ptr[id];
                        break;
                    case 1: // add模式
                        output_ptr[offset] += upd_ptr[id];
                        break;
                    case 2: // mul模式
                        output_ptr[offset] *= upd_ptr[id];
                        break;
                    case 3: // min模式
                        output_ptr[offset] =
                            output_ptr[offset] > upd_ptr[id] ? upd_ptr[id] : output_ptr[offset];
                        break;
                    case 4: // max模式
                        output_ptr[offset] =
                            output_ptr[offset] < upd_ptr[id] ? upd_ptr[id] : output_ptr[offset];
                        break;
                    }
                }
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx