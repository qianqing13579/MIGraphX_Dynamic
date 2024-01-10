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
#include <migraphx/gpu/multibroadcast_dynamic.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_multibroadcast_dynamic::compute_shape(std::vector<shape> inputs) const
{
    return op.compute_shape(inputs);
}

static shape compute_shape_dynamic(const shape& input, const std::vector<std::size_t>& output_lens)
{
    auto t = input.type();

    if(input.lens().empty())
    {
        MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should be > 0");
    }

    if(input.lens().size() > output_lens.size())
    {
        MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should <= output size");
    }

    auto offset = output_lens.size() - input.lens().size();
    for(std::ptrdiff_t i = input.lens().size() - 1; i >= 0; i--)
    {
        if(output_lens[i + offset] != input.lens()[i] and input.lens()[i] != 1)
        {
            MIGRAPHX_THROW("MULTIBROADCAST: input shape {" + to_string_range(input.lens()) +
                           "} cannot be broadcasted to {" + to_string_range(output_lens) + "}!");
        }
    }

    std::vector<size_t> bcast_strides(output_lens.size(), 0);
    for(std::ptrdiff_t i = input.lens().size() - 1; i >= 0; i--)
    {
        if(output_lens[i + offset] == input.lens()[i])
        {
            bcast_strides[i + offset] = input.strides()[i];
        }
    }
    return {t, output_lens, bcast_strides};
}

argument hip_multibroadcast_dynamic::compute(context& ctx,
                                             const shape& output_shape,
                                             const std::vector<argument>& args) const
{
    if(op.is_const == 0)
    {

        // 计算dim，shape tensor在FP16模式下还是float类型
        std::vector<float> shape_data(args[1].get_shape().elements());

        // 拷贝到gpu
        hipMemcpyAsync(shape_data.data(),
                       args[1].data(),
                       args[1].get_shape().bytes(),
                       hipMemcpyDeviceToHost,
                       ctx.get_stream().get());

        ctx.finish();

        // 重新计算输出shape
        std::vector<std::size_t> shape_data2(args[1].get_shape().elements());
        for(int i = 0; i < shape_data.size(); ++i)
        {
            shape_data2[i] = static_cast<std::size_t>(shape_data[i]);
        }
        auto out_lens   = compute_broadcasted_lens(args[0].get_shape().lens(), shape_data2);
        shape new_shape = compute_shape_dynamic(args[0].get_shape(), out_lens);

        return args[0].reshape(new_shape);
    }
    else
    {
        auto out_lens   = compute_broadcasted_lens(args[0].get_shape().lens(), op.dims);
        shape new_shape = compute_shape_dynamic(args[0].get_shape(), out_lens);

        return args[0].reshape(new_shape);
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
