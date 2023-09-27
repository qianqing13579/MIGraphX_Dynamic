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
#include <migraphx/gpu/tile.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/tile.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static shape compute_shape_tile(const shape& input,const std::vector<int64_t>& repeats)
{
    auto&& idims = input.lens();
    std::vector<std::size_t> rdims(idims.begin(), idims.end());
    std::size_t num_dims = rdims.size();
    for(std::size_t i = 0; i < num_dims; i++)
    {
        rdims[i] *= repeats[i];
    }

    shape s{input.type(), rdims};
    return s;
    
}

shape hip_tile::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_tile::compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
{
    std::vector<int64_t> repeats;

    // 获取repeats
    std::vector<float> shape_data(args[1].get_shape().elements()); // shape tensor在FP16模式下还是float类型

    // 拷贝到gpu
    hipMemcpyAsync(shape_data.data(), args[1].data(), args[1].get_shape().bytes(), hipMemcpyDeviceToHost,ctx.get_stream().get());
    ctx.finish();
    
    repeats.resize(args[1].get_shape().elements());
    for(int i=0;i<shape_data.size();++i)
    {
        repeats[i]=static_cast<int64_t>(shape_data[i]);
    }

    shape new_shape=compute_shape_tile(args.front().get_shape(), repeats);
    argument result=args.back();
    result=result.reshape(new_shape);

    return device::tile(ctx.get_stream().get(), result, args[0], repeats);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
