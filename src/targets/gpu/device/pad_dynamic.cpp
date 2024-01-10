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
#include <migraphx/clamp.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/pad.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/float_equal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument pad_dynamic(hipStream_t stream,
                     argument result,
                     argument arg1,
                     float value,
                     std::vector<std::int64_t> pads,
                     int mode,
                     size_t pad_ndims,
                     bool PadHW)
{

    // 获取输入输出大小
    migraphx::shape output_shape           = result.get_shape();
    migraphx::shape input_shape            = arg1.get_shape();
    std::vector<std::size_t> input_dims    = arg1.get_shape().lens();
    std::vector<std::size_t> input_strides = arg1.get_shape().strides();
    std::vector<int> ldims(pads.begin(), pads.begin() + pad_ndims);

    if(PadHW)
    {
        int height_dim = 2;
        int width_dim  = 3;

        if(pad_ndims == 3)
        {
            height_dim = 1;
            width_dim  = 2;
        }
        else if(pad_ndims == 2)
        {
            height_dim = 0;
            width_dim  = 1;
        }

        int output_height    = output_shape.lens()[height_dim];
        int output_width     = output_shape.lens()[width_dim];
        int input_height     = input_shape.lens()[height_dim];
        int input_width      = input_shape.lens()[width_dim];
        int pad_height_start = ldims[height_dim];
        int pad_width_start  = ldims[width_dim];

        hip_visit_all(result, arg1)([&](auto output, auto input) {
            using type      = typename decltype(output)::value_type;
            type device_val = pad_clamp<host_type<type>>(value);

            gs_launch(stream, output_shape.elements())([=](auto i, auto idx) __device__ {
                int w  = i % output_width;
                int h  = (i / output_width) % output_height;
                int nc = i / output_width / output_height;

                int current_input_width  = w - pad_width_start;
                int current_input_height = h - pad_height_start;

                switch(mode)
                {
                case 0: // constant模式
                    output[i] =
                        (current_input_height < 0 || current_input_width < 0 ||
                         current_input_height >= input_height || current_input_width >= input_width)
                            ? device_val
                            : input[(nc * input_height + current_input_height) * input_width +
                                    current_input_width];
                    break;
                case 1: // reflect模式
                    current_input_height = std::max(current_input_height, -current_input_height);
                    current_input_height =
                        std::min(static_cast<int>(current_input_height),
                                 2 * static_cast<int>(input_height) - current_input_height - 2);
                    current_input_width = std::max(current_input_width, -current_input_width);
                    current_input_width =
                        std::min(static_cast<int>(current_input_width),
                                 2 * static_cast<int>(input_width) - current_input_width - 2);
                    output[i] = input[(nc * input_height + current_input_height) * input_width +
                                      current_input_width];
                    break;
                case 2: // edge模式
                    current_input_height = std::max(
                        0, std::min(current_input_height, static_cast<int>(input_height - 1)));
                    current_input_width = std::max(
                        0, std::min(current_input_width, static_cast<int>(input_width - 1)));
                    output[i] = input[(nc * input_height + current_input_height) * input_width +
                                      current_input_width];
                    break;
                }
            });
        });
    }
    else
    {
        hip_visit_all(result, arg1)([&](auto output, auto input) {
            using hip_index = typename decltype(output)::hip_index;
            using type      = typename decltype(output)::value_type;
            type device_val = pad_clamp<host_type<type>>(value);

            hip_index offsets, input_dim, input_stride;
            std::copy(ldims.begin(), ldims.begin() + offsets.size(), offsets.begin());
            std::copy(input_dims.begin(), input_dims.begin() + input_dim.size(), input_dim.begin());
            std::copy(input_strides.begin(),
                      input_strides.begin() + input_stride.size(),
                      input_stride.begin());

            // 启动kernel，计算对应索引
            gs_launch(stream, output_shape.elements())([=](auto i) __device__ {
                bool use_pad_value = false;
                int in_coord       = 0;
                int input_index    = 0;
                auto out_coord     = output.get_shape().multi(i); // 计算出多维索引

                for(int64_t dim = 0; dim < pad_ndims && !use_pad_value; ++dim)
                {
                    if(out_coord[dim] < offsets[dim])
                    {
                        switch(mode)
                        {
                        case 0: use_pad_value = true; break;
                        case 1: in_coord = offsets[dim] - out_coord[dim]; break;
                        case 2: in_coord = 0; break;
                        }
                    }
                    else if(out_coord[dim] >= offsets[dim] + input_dim[dim])
                    {
                        switch(mode)
                        {
                        case 0: use_pad_value = true; break;
                        case 1:
                            in_coord = input_dim[dim] - 2 -
                                       (out_coord[dim] - (offsets[dim] + input_dim[dim]));
                            break;
                        case 2: in_coord = input_dim[dim] - 1; break;
                        }
                    }
                    else
                    {
                        in_coord = out_coord[dim] - offsets[dim];
                    }
                    input_index += input_stride[dim] * in_coord;
                }
                output[i] = use_pad_value ? device_val : input[input_index];
            });
        });
    }

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx