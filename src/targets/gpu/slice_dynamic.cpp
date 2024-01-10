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
#include <migraphx/gpu/slice_dynamic.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_slice_dynamic::normalize_compute_shape(std::vector<shape> inputs) const
{
    return op.normalize_compute_shape(inputs);
}

static auto fix_index(const std::vector<std::size_t>& lens, std::size_t axis, int64_t index)
{
    int64_t r = std::min(index, static_cast<int64_t>(lens[axis]));
    if(r < 0)
        r += lens[axis];
    return std::size_t(r);
}

static auto compute_offset(const shape& s,
                           std::vector<int64_t>& starts,
                           std::vector<int64_t>& ends,
                           std::vector<int64_t> axes)
{
    const std::vector<std::size_t>& lens    = s.lens();
    const std::vector<std::size_t>& strides = s.strides();
    auto offset                             = 0;
    if(not axes.empty())
    {
        for(std::size_t i = 0; i < axes.size(); i++)
        {
            auto axis = axes[i];
            offset += fix_index(lens, axis, starts[i]) * strides[axis];
        }
    }
    else
    {
        for(std::size_t axis = 0; axis < lens.size(); axis++)
        {
            offset += fix_index(lens, axis, starts[axis]) * strides[axis];
        }
    }
    return offset;
}

static shape compute_shape_dynamic(const shape& input_shape,
                                   std::vector<int64_t>& starts,
                                   std::vector<int64_t>& ends,
                                   std::vector<int64_t> axes)
{
    auto t                  = input_shape.type();
    const auto& old_lens    = input_shape.lens();
    const auto& old_strides = input_shape.strides();

    if(std::any_of(
           axes.begin(), axes.end(), [&](auto i) { return (i >= old_lens.size() and i < 0); }))
    {
        MIGRAPHX_THROW("SLICE: input axis " + to_string_range(axes) + " out of range");
    }

    if(starts.size() != axes.size() or axes.size() != ends.size())
    {
        MIGRAPHX_THROW("SLICE: inconsistent sizes");
    }

    std::vector<std::size_t> new_lens = old_lens;
    for(std::size_t i = 0; i < axes.size(); i++)
    {
        auto axis      = axes[i];
        new_lens[axis] = fix_index(old_lens, axis, ends[i]) - fix_index(old_lens, axis, starts[i]);
    }
    return shape{t, new_lens, old_strides};
}

/*

注意：在老版本的ONNX标准中，slice的starts,ends和axes是作为属性，而不是作为输入参数
args[0]: input tensor
args[1]: std::vector<int64_t> starts
args[2]: std::vector<int64_t> ends
args[3]: std::vector<int64_t> axes

*/

argument hip_slice_dynamic::compute(context& ctx,
                                    const shape& output_shape,
                                    const std::vector<argument>& args) const
{
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
    std::vector<int64_t> axes;

    // 新标准
    if(args.size() > 1)
    {
        // 拷贝数据
        std::vector<float> shape_data_starts(
            args[1].get_shape().elements()); // shape tensor在FP16模式下还是float类型
        std::vector<float> shape_data_ends(args[2].get_shape().elements());
        if(op.is_const_stars == 0)
        {
            // 拷贝到gpu
            hipMemcpyAsync(shape_data_starts.data(),
                           args[1].data(),
                           args[1].get_shape().bytes(),
                           hipMemcpyDeviceToHost,
                           ctx.get_stream().get());
        }
        if(op.is_const_ends == 0)
        {
            // 拷贝到gpu
            hipMemcpyAsync(shape_data_ends.data(),
                           args[2].data(),
                           args[2].get_shape().bytes(),
                           hipMemcpyDeviceToHost,
                           ctx.get_stream().get());
        }
        if(op.is_const_stars == 0 || op.is_const_ends == 0)
        {
            ctx.finish();
        }

        // 计算starts
        if(op.is_const_stars == 0)
        {
            starts.resize(args[1].get_shape().elements());
            for(int i = 0; i < shape_data_starts.size(); ++i)
            {
                starts[i] = static_cast<int64_t>(shape_data_starts[i]);
            }
        }
        else
        {
            starts = op.starts;
        }

        // 计算ends
        if(op.is_const_ends == 0)
        {
            ends.resize(args[2].get_shape().elements());
            for(int i = 0; i < shape_data_ends.size(); ++i)
            {
                ends[i] = static_cast<int64_t>(shape_data_ends[i]);
            }
        }
        else
        {
            ends = op.ends;
        }

        // 计算axes
        {
            axes = op.axes;
        }
    }
    // 旧标准中，直接作为属性
    else
    {
        starts = op.starts;
        ends   = op.ends;
        axes   = op.axes;
    }

    // 执行normalize，当end是int64_t的最大值的时候,转换为float会变成负值，此时需要normalize
    bool need_normalize = std::any_of(ends.begin(), ends.end(), [=](int64_t i) { return i < 0; });
    if(need_normalize)
    {
        migraphx::operation tuned_op = *this;
        normalize_attributes(tuned_op, args[0].get_shape().max_lens());
        starts = any_cast<gpu::hip_slice_dynamic>(tuned_op).op.starts;
        ends   = any_cast<gpu::hip_slice_dynamic>(tuned_op).op.ends;
        axes   = any_cast<gpu::hip_slice_dynamic>(tuned_op).op.axes;
    }

    // 计算输出shape
    shape new_shape = compute_shape_dynamic(args[0].get_shape(), starts, ends, axes);

    auto input  = args[0];
    auto offset = compute_offset(input.get_shape(), starts, ends, axes) * new_shape.type_size();
    return {std::move(new_shape), [=] { return input.data() + offset; }};
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
