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
#include <iterator>
#include <migraphx/simplify_shape_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/as_shape.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <unordered_set>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/*
main:@57 = gpu::Shape[max_shape=64](main:@47,main:@56) -> int64_type, {1}, {1}
main:@59 = gpu::convert[target_type=2](main:@57,main:@58) -> float_type, {1}, {1}
main:@60 = gpu::multibroadcast_dynamic[max_out_lens={64}](main:@48,main:@59) -> float_type, {64}, {0}
*/
struct find_shape_convert
{
    auto matcher() const 
    { 
        return match::name("multibroadcast_dynamic")(
            match::either_arg(0, 1)(
                match::name("convert")(match::used_once,match::arg(0)(match::name("Shape"))),match::any())
                ); 
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto multibroadcast   = mr.result;
        auto convert     = multibroadcast->inputs()[1];
        auto shape     = convert->inputs()[0];
        m.replace_instruction(convert, shape);
    }
};

void simplify_shape_op::apply(module& m) const
{
    match::find_matches(m,
                        find_shape_convert{}// 优化动态shape中Shape->convert->multibroadcast_dynamic
                        );
    dead_code_elimination{}.apply(m);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
