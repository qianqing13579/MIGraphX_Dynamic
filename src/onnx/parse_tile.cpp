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

struct parse_tile : op_parser<parse_tile>
{
    std::vector<op_desc> operators() const { return {{"Tile"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {

        if(info.mod->get_dynamic())
        {
            std::vector<int64_t> repeats{};
            auto repeats_arg = args.at(1)->eval_for_shape();
            check_arg_empty(repeats_arg, "PARSE_tile: repeats input must be constant");
            repeats_arg.visit([&](auto v) { repeats.assign(v.begin(), v.end()); });

            auto input_shape        = args[0]->get_shape();
            auto input_shape_lens   = input_shape.lens();
            auto repeats_shape      = args[1]->get_shape();
            auto repeats_shape_lens = repeats_shape.lens();

            if(repeats_shape_lens.size() != 1)
            {
                MIGRAPHX_THROW("PARSE_Tile: 'repeat' input tensor must be 1 dimensional");
            }

            if(repeats_shape.elements() != input_shape_lens.size())
            {
                MIGRAPHX_THROW("PARSE_Tile: 'repeat' input tensor must have the same length as the "
                               "'input' tensor");
            }

            return info.add_instruction(migraphx::make_op("tile", {{"max_repeats", repeats}}),
                                        args);
        }
        else
        {
            migraphx::argument arg_s = args[1]->eval();
            check_arg_empty(arg_s, "PARSE_TILE: dynamic shape is not supported");
            std::vector<std::int64_t> repeats;
            arg_s.visit([&](auto input) { repeats.assign(input.begin(), input.end()); });

            auto l0 = args[0];
            for(int i = 0; i < repeats.size(); i++)
            {
                auto l1 = l0;
                for(int j = 1; j < repeats[i]; j++)
                {
                    l0 = info.add_instruction(make_op("concat", {{"axis", i}}), l0, l1);
                }
            }
            return l0;
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
