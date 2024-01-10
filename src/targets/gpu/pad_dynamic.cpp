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
#include <migraphx/gpu/pad_dynamic.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/pad_dynamic.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_pad_dynamic::compute_shape(std::vector<shape> inputs) const
{
    return op.compute_shape({inputs[0]});
}

static shape compute_shape_dynamic(const shape& input, const std::vector<int64_t>& pads)
{
    auto&& idims = input.lens();
    std::vector<std::size_t> rdims(idims.begin(), idims.end());
    std::size_t num_dims = rdims.size();

    for(std::size_t i = 0; i < num_dims; i++)
    {
        rdims[i] += pads[i] + pads[i + num_dims];
    }

    shape s{input.type(), rdims};
    return s;
}

bool IsNCHWInputWithPaddingAlongHAndW(const std::vector<int64_t> pads, const size_t pad_ndims)
{
    std::vector<int> ldims(pads.begin(), pads.begin() + pad_ndims);
    std::vector<int> rdims(pads.begin() + pad_ndims, pads.end());
    assert(ldims.size() == rdims.size());

    if(pad_ndims == 2)
    {
        return true;
    }

    if(pad_ndims == 3 && ldims[0] == 0 && rdims[0] == 0)
    {
        return true;
    }

    if(pad_ndims == 4 && ldims[0] == 0 && ldims[1] == 0 && rdims[0] == 0 && rdims[1] == 0)
    {
        return true;
    }
    return false;
}

argument hip_pad_dynamic::compute(context& ctx,
                                  const shape& output_shape,
                                  const std::vector<argument>& args) const
{

    // 获取pads
    std::vector<int64_t> pads = op.pads;

    // 获取pad_ndims
    size_t pad_ndims = op.pad_ndims();

    if(args.size() > 2)
    {
        std::vector<float> shape_data(args[1].get_shape().elements());

        // 拷贝到gpu
        hipMemcpyAsync(shape_data.data(),
                       args[1].data(),
                       args[1].get_shape().bytes(),
                       hipMemcpyDeviceToHost,
                       ctx.get_stream().get());

        ctx.finish();

        for(int i = 0; i < shape_data.size(); ++i)
        {
            pads[i] = static_cast<int64_t>(shape_data[i]);
        }
    }

    // 重新计算输出shape
    shape new_shape = compute_shape_dynamic(args.front().get_shape(), pads);

    argument result = args.back();
    result          = result.reshape(new_shape);

    bool PadHW = IsNCHWInputWithPaddingAlongHAndW(pads, pad_ndims);

    return device::pad_dynamic(
        ctx.get_stream().get(), result, args.front(), op.value, op.pads, op.mode, pad_ndims, PadHW);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
