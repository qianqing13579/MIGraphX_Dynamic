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
#ifndef MIGRAPHX_GUARD_OPERATORS_RANGE_HPP
#define MIGRAPHX_GUARD_OPERATORS_RANGE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct range
{
    float start;
    float limit;
    float delta;
    int is_const_start=0;
    int is_const_limit=0;
    int is_const_delta=0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.start, "max_start"), 
                    f(self.limit, "max_limit"),
                    f(self.delta, "delta"),
                    f(self.is_const_start, "is_const_start"),
                    f(self.is_const_limit, "is_const_limit"),
                    f(self.is_const_delta, "is_const_delta"));
    }

    std::string name() const { return "range"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).same_type();

        size_t num_elements = static_cast<size_t>(ceil(static_cast<double>(limit - start) / static_cast<double>(delta)));
        
        return shape{inputs[0].type(), {num_elements}};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {

        argument start_arg=args[0];
        argument limit_arg=args[1];
        argument delta_arg=args[2];

        argument result;
        visit_all(start_arg, limit_arg, delta_arg)([&](auto start, auto limit, auto delta) {
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

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
