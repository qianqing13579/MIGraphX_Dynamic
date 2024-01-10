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
#ifndef MIGRAPHX_GUARD_OPERATORS_SHAPE_OP_HPP
#define MIGRAPHX_GUARD_OPERATORS_SHAPE_OP_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// 这里不能直接用shape,因为migraphx中已经有shape类型了
struct Shape
{
    literal max_shape; // 最大shape

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.max_shape, "max_shape"));
    }

    std::string name() const { return "Shape"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);

        std::vector<std::size_t> arg_shape = inputs[0].lens();

        return {migraphx::shape::int64_type, {arg_shape.size()}};
    }
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        std::vector<std::size_t> arg_shape = args[0].get_shape().lens();
        migraphx::shape s(migraphx::shape::int64_type, {arg_shape.size()});
        migraphx::argument data = migraphx::generate_argument(s);
        std::transform(arg_shape.begin(), arg_shape.end(), (int64_t*)data.data(), [](auto i) {
            return int64_t(i);
        });

        return data;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
