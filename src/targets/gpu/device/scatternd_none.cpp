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
#include <migraphx/gpu/device/scatternd_none.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument scatternd_none(hipStream_t stream, argument result, std::vector<argument> args)
{
    // 获取indice 中的最后一个维度
    migraphx::shape indices_shape = args[1].get_shape();
    int64_t last_index_dimension  = indices_shape.lens()[indices_shape.lens().size() - 1];

    // 获取input的维度信息
    migraphx::shape input_shape  = args[0].get_shape();
    int64_t num_updates_elements = 1;
    if(last_index_dimension < input_shape.lens().size())
    {
        for(int64_t i = last_index_dimension; i < input_shape.lens().size(); i++)
        {
            num_updates_elements *= input_shape.lens()[i];
        }
    }

    std::vector<std::size_t> input_strides = args[0].get_shape().strides();
    std::vector<int64_t> input_stride(last_index_dimension, 0);
    std::vector<int64_t> input_shapes(last_index_dimension, 0);
    for(int64_t i = 0; i < last_index_dimension; ++i)
    {
        input_stride[i] = input_strides[i];
        input_shapes[i] = input_shape.lens()[i];
    }

    hip_visit_all(result, args[0], args[2])([&](auto output, auto data, auto update) {
        auto* output_ptr     = device_cast(output.data());
        const auto* data_ptr = device_cast(data.data());
        auto* update_ptr     = device_cast(update.data());

        gs_launch(stream, input_shape.elements())(
            [=](auto i) __device__ { output_ptr[i] = data_ptr[i]; }); // 将输入数据拷贝给输出数据

        hip_visit_all(args[1])([&](auto indices) {
            const auto* indices_ptr = device_cast(indices.data());
            using hip_index         = typename decltype(output)::hip_index;
            hip_index input_stride_1, input_shapes_1;
            std::copy(input_stride.begin(), input_stride.end(), input_stride_1.begin());
            std::copy(input_shapes.begin(), input_shapes.end(), input_shapes_1.begin());

            gs_launch(stream, args[1].get_shape().elements() / last_index_dimension)(
                [=](auto id) __device__ {
                    int64_t data_offset  = 0;
                    size_t indices_start = last_index_dimension * id;
                    size_t indices_end   = indices_start + last_index_dimension;

                    for(size_t i = indices_start; i < indices_end; ++i)
                    {
                        int64_t index             = indices_ptr[i];
                        int64_t element_count_dim = input_stride_1[i - indices_start];
                        int64_t dim_value         = input_shapes_1[i - indices_start];

                        // 边界控制，将index值限定在 index >= -dim_value && index < dim_value
                        if(index >= 0)
                        {
                            if(index >= dim_value)
                            {
                                index = dim_value - 1;
                            }
                        }
                        else
                        {
                            if(index < -dim_value)
                            {
                                index = 0;
                            }
                            else
                            {
                                index += dim_value;
                            }
                        }
                        data_offset += (index * element_count_dim);
                    }

                    auto* update_data_base = update_ptr + num_updates_elements * id;
                    auto* output_data_base = output_ptr + data_offset;

                    for(size_t i = 0; i < num_updates_elements; ++i)
                    {
                        output_data_base[i] = update_data_base[i];
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