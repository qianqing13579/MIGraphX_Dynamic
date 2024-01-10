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
#include <migraphx/gpu/Shape.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_Shape::compute_shape(const std::vector<shape>& inputs) const
{
    return op.compute_shape({inputs.at(0)});
}

argument hip_Shape::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    std::vector<std::size_t> arg_shape = args.front().get_shape().lens();
    migraphx::shape s{migraphx::shape::int64_type, {arg_shape.size()}};
    migraphx::argument data{s};
    std::transform(arg_shape.begin(), arg_shape.end(), (int64_t*)data.data(), [](auto i) {
        return int64_t(i);
    });

    // 拷贝到gpu
    hipMemcpyAsync(args.back().data(),
                   data.data(),
                   data.get_shape().bytes(),
                   hipMemcpyHostToDevice,
                   ctx.get_stream().get());

    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
