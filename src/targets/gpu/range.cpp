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
#include <migraphx/gpu/range.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_range::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_range::compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
{
    // 计算start
    argument start_arg{args[0].get_shape()};
    if(op.is_const_start==1)
    {
        float start=op.start;
        start_arg.visit([&](auto data) { data[0]=start;});

    }
    else
    {
        hipMemcpyAsync(start_arg.data(), args[0].data(), args[0].get_shape().bytes(), hipMemcpyDeviceToHost,ctx.get_stream().get());
    }
    
    // 计算limit
    argument limit_arg{args[1].get_shape()};
    if(op.is_const_limit==1)
    {
        float limit=op.limit;
        limit_arg.visit([&](auto data) { data[0]=limit;});
    }
    else
    {
        hipMemcpyAsync(limit_arg.data(), args[1].data(), args[1].get_shape().bytes(), hipMemcpyDeviceToHost,ctx.get_stream().get());
    }

    // 计算delta
    argument delta_arg{args[2].get_shape()};
    if(op.is_const_delta==1)
    {
        float delta=op.delta;
        delta_arg.visit([&](auto data) { data[0]=delta;});
    }
    else
    {
        hipMemcpyAsync(delta_arg.data(), args[2].data(), args[2].get_shape().bytes(), hipMemcpyDeviceToHost,ctx.get_stream().get());
    }

    if(op.is_const_start==0||op.is_const_limit==0||op.is_const_delta==0)
    {
        ctx.finish();
    }
    
    // 生成最后的结果
    argument result;
    visit_all(start_arg, limit_arg, delta_arg)([&](auto start, auto limit, auto delta) 
            {
                auto start_val = start.front();
                auto limit_val = limit.front();
                auto delta_val = delta.front();

                size_t num_elements = static_cast<size_t>(
                    ceil(static_cast<double>(limit_val - start_val) / static_cast<double>(delta_val)));

                assert(num_elements > 0);

                using type = decltype(start_val);

                std::vector<type> range_vals(num_elements);

                std::generate(range_vals.begin(), range_vals.end(), [&]() {
                    auto result = start_val;
                    start_val += delta_val;
                    return result;
                });

                result= literal{shape{args[0].get_shape().type(), {num_elements}}, range_vals}.get_argument();
            });

    // 拷贝到最后的输出tensor
    argument output_tensor=args[args.size()-1].reshape(result.get_shape());
    hipMemcpyAsync(output_tensor.data(), result.data(), result.get_shape().bytes(), hipMemcpyHostToDevice,ctx.get_stream().get());

    return output_tensor;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
