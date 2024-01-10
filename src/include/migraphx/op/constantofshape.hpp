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
#ifndef MIGRAPHX_GUARD_OPERATORS_CONSTANTOFSHAPE_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONSTANTOFSHAPE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct constantofshape
{
    shape output_shape;
    literal value;
    int is_const = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_shape, "max_shape"),
                    f(self.value, "value"),
                    f(self.is_const, "is_const"));
    }

    std::string name() const { return "constantofshape"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();

        return output_shape;
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        // input is empty, output is a scalar
        auto type = value.get_shape().type();

        if(args.empty())
        {
            MIGRAPHX_THROW("ConstantOfShape : must have 1 input!");
        }
        else
        {
            migraphx::shape s;
            // empty input tensor, output is a scalar
            if(args[0].get_shape().elements() == 0)
            {
                s = migraphx::shape{type, {1}, {0}};
            }
            else
            {
                migraphx::argument in = args[0];

                std::vector<std::size_t> dims;
                in.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
                s = migraphx::shape{type, dims};
            }

            migraphx::argument result{};
            value.visit([&](auto val) {
                using val_type = std::remove_cv_t<typename decltype(val)::value_type>;
                // l_val contains only one element
                std::vector<val_type> out_vec(s.elements(), val.front());
                result = migraphx::generate_argument(s);
                std::transform(out_vec.begin(),
                               out_vec.end(),
                               (val_type*)result.data(),
                               [](auto i) { return val_type(i); });
            });

            return result;
        }
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
