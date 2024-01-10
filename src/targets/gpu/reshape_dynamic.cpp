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
#include <migraphx/gpu/reshape_dynamic.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_reshape_dynamic::compute_shape(std::vector<shape> inputs) const
{
    return op.compute_shape(inputs);
}

static shape compute_shape_dynamic(const shape& input, const std::vector<int64_t>& dims)
{
    auto&& idims = input.lens();
    std::vector<std::size_t> rdims(dims.begin(), dims.end());
    auto n_neg_dims = std::count(dims.begin(), dims.end(), -1);
    if(n_neg_dims > 1)
        MIGRAPHX_THROW("Reshape: Dimensions for reshape can only have one -1 dim");

    for(std::size_t i = 0; i < dims.size(); i++)
    {
        if(dims[i] == 0)
            rdims[i] = idims[i];

        // since rdims using size_t type, -1 is the max value
        // is size_t that cause later compuation incorrect
        if(dims[i] == -1)
            rdims[i] = 1;
    }

    if(n_neg_dims > 0)
    {
        size_t missing_dim =
            input.elements() /
            std::accumulate(rdims.begin(), rdims.end(), 1, std::multiplies<int64_t>());
        for(std::size_t i = 0; i < rdims.size(); i++)
        {
            if(dims[i] == -1)
                rdims[i] = missing_dim;
        }
    }
    shape s{input.type(), rdims};
    if(s.elements() != input.elements())
        MIGRAPHX_THROW("Reshape: Wrong number of elements for reshape: reshape has " +
                       std::to_string(s.elements()) + " elements whereas the input has " +
                       std::to_string(input.elements()));
    return s;
}

argument hip_reshape_dynamic::compute(context& ctx,
                                      const shape& output_shape,
                                      const std::vector<argument>& args) const
{

    if(op.is_const == 1)
    {
        shape new_shape = compute_shape_dynamic(args.front().get_shape(), op.dims);
        return args[0].reshape(new_shape);
    }
    else
    {
        // 计算dim，shape tensor在FP16模式下还是float类型
        std::vector<float> shape_data(args.back().get_shape().elements());

        // 拷贝到gpu
        hipMemcpyAsync(shape_data.data(),
                       args.back().data(),
                       args.back().get_shape().bytes(),
                       hipMemcpyDeviceToHost,
                       ctx.get_stream().get());

        ctx.finish();

        std::vector<int64_t> shape_data2(args.back().get_shape().elements());
        for(int i = 0; i < shape_data.size(); ++i)
        {
            shape_data2[i] = static_cast<int64_t>(shape_data[i]);
        }

        // 重新计算输出shape
        shape new_shape = compute_shape_dynamic(args.front().get_shape(), shape_data2);

        return args[0].reshape(new_shape);
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
