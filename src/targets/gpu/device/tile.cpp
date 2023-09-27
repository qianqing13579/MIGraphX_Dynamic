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
#include <migraphx/gpu/device/tile.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument tile(hipStream_t stream, argument result, argument arg1, std::vector<int64_t> repeats)
{
    const auto& input_shape = arg1.get_shape();
    auto lens               = input_shape.lens();
    auto rank               = lens.size();
    shape out_comp_shape = result.get_shape();
    auto output_lens = out_comp_shape.lens();
    auto input_strides = input_shape.strides();
    auto output_strides = out_comp_shape.strides(); 
    std::size_t nelements = result.get_shape().elements();

    visit_all(result, arg1)([&](auto output, auto input_v) {
        hip_visit_views(input_v, out_comp_shape)([&](auto input, auto out_comp) {
            auto* output_ptr        = device_cast(output.data());
            using hip_index = typename decltype(input)::hip_index;
            hip_index input_strides_d,output_strides_d,lens_d;
            std::copy(input_strides.begin(), input_strides.end(), input_strides_d.begin());
            std::copy(output_strides.begin(), output_strides.end(), output_strides_d.begin());
            std::copy(lens.begin(), lens.end(), lens_d.begin());

            if (out_comp_shape == input_shape)
            {
                gs_launch(stream, nelements, 256)([=](auto i) __device__ {
                    output_ptr[i] = input[i];
                });
            }
            else
            {
                gs_launch(stream, nelements, 256)([=](auto i) __device__ {
                    if (i < nelements)
                    {
                        int offset = i;
                        int input_index = 0;
                        for (auto dim = 0; dim < output_strides_d.size(); ++dim)
                        {
                            int out_coord = offset / output_strides_d[dim];
                            int r = offset % output_strides_d[dim];
                            int in_coord = out_coord % lens_d[dim];
                            input_index += input_strides_d[dim] * in_coord;
                            offset = r;
                        }
                        output_ptr[i] = input[input_index];
                    }
                    
                });
            }
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
