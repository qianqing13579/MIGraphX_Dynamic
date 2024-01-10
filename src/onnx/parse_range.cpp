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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_range : op_parser<parse_range>
{
    std::vector<op_desc> operators() const { return {{"Range"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        // 动态实现
        if(info.mod->get_dynamic())
        {
            int is_const_start = 0;
            int is_const_limit = 0;
            int is_const_delta = 0;

            auto start_arg = args[0]->eval_for_shape();
            check_arg_empty(start_arg, "PARSE_RANGE: start arg dynamic shape is not supported");
            if(args[0]->name() == "@literal")
            {
                is_const_start = 1;
            }

            auto limit_arg = args[1]->eval_for_shape();
            check_arg_empty(limit_arg, "PARSE_RANGE: limit arg dynamic shape is not supported");
            if(args[1]->name() == "@literal")
            {
                is_const_limit = 1;
            }

            auto delta_arg = args[2]->eval_for_shape();
            check_arg_empty(delta_arg, "PARSE_RANGE: delta arg dynamic shape is not supported");
            if(args[2]->name() == "@literal")
            {
                is_const_delta = 1;
            }

            assert(args[0]->get_shape().elements() == 1 and args[1]->get_shape().elements() == 1 and
                   args[2]->get_shape().elements() == 1);

            instruction_ref l0;

            visit_all(start_arg, limit_arg, delta_arg)([&](auto start, auto limit, auto delta) {
                float start_val = (float)start.front();
                float limit_val = (float)limit.front();
                float delta_val = (float)delta.front();

                l0 = info.add_instruction(make_op("range",
                                                  {{"max_start", start_val},
                                                   {"max_limit", limit_val},
                                                   {"delta", delta_val},
                                                   {"is_const_start", is_const_start},
                                                   {"is_const_limit", is_const_limit},
                                                   {"is_const_delta", is_const_delta}}),
                                          args);
            });
            return l0;
        }
        else
        {
            auto start_arg = args[0]->eval();
            check_arg_empty(start_arg, "PARSE_RANGE: start arg dynamic shape is not supported");
            auto limit_arg = args[1]->eval();
            check_arg_empty(limit_arg, "PARSE_RANGE: limit arg dynamic shape is not supported");
            auto delta_arg = args[2]->eval();
            check_arg_empty(delta_arg, "PARSE_RANGE: delta arg dynamic shape is not supported");

            assert(args[0]->get_shape().elements() == 1 and args[1]->get_shape().elements() == 1 and
                   args[2]->get_shape().elements() == 1);

            instruction_ref l0;

            visit_all(start_arg, limit_arg, delta_arg)([&](auto start, auto limit, auto delta) {
                auto start_val = start.front();
                auto limit_val = limit.front();
                auto delta_val = delta.front();

                size_t num_elements = static_cast<size_t>(ceil(
                    static_cast<double>(limit_val - start_val) / static_cast<double>(delta_val)));

                assert(num_elements > 0);

                using type = decltype(start_val);

                std::vector<type> range_vals(num_elements);

                std::generate(range_vals.begin(), range_vals.end(), [&]() {
                    auto result = start_val;
                    start_val += delta_val;
                    return result;
                });

                l0 = info.add_literal(
                    {shape{args[0]->get_shape().type(), {num_elements}}, range_vals});
            });
            return l0;
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
