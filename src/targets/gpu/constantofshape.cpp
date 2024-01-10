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
#include <migraphx/gpu/constantofshape.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_constantofshape::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_constantofshape::compute(context& ctx,
                                      const shape& output_shape,
                                      const std::vector<argument>& args) const
{
    // 重新生成常量
    auto type = op.value.get_shape().type();
    migraphx::shape s;
    if(op.is_const == 0)
    {
        // 计算dim，shape tensor在FP16模式下还是float类型
        std::vector<float> shape_data(args[0].get_shape().elements());

        // 拷贝到gpu
        hipMemcpyAsync(shape_data.data(),
                       args[0].data(),
                       args[0].get_shape().bytes(),
                       hipMemcpyDeviceToHost,
                       ctx.get_stream().get());

        ctx.finish();

        // 重新计算输出shape
        std::vector<std::size_t> shape_data2(args[0].get_shape().elements());
        for(int i = 0; i < shape_data.size(); ++i)
        {
            shape_data2[i] = static_cast<std::size_t>(shape_data[i]);
        }

        if(shape_data2.size() == 0) // 如果为空，则是一个标量
        {
            s = migraphx::shape{type, {1}, {0}};
        }
        else
        {
            s = migraphx::shape{type, shape_data2};
        }
    }
    else
    {
        s = op.output_shape;
    }

    migraphx::argument result{s};
    op.value.visit([&](auto val) {
        auto data       = val.front();
        using data_type = decltype(data);
        // using data_type = std::remove_cv_t<typename decltype(val)::value_type>;
        for(int i = 0; i < s.elements(); ++i)
        {
            ((data_type*)result.data())[i] = data;
        }
    });

    // 拷贝到输出tensor
    argument output_tensor = args[args.size() - 1].reshape(s);
    hipMemcpyAsync(output_tensor.data(),
                   result.data(),
                   result.get_shape().bytes(),
                   hipMemcpyHostToDevice,
                   ctx.get_stream().get());

    return output_tensor;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
