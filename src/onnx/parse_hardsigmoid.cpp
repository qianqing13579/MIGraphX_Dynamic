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
#include <migraphx/op/Shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_hardsigmoid : op_parser<parse_hardsigmoid>
{
    std::vector<op_desc> operators() const { return {{"HardSigmoid"}, {"HardSwish"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        // 动态实现
        if(info.mod->get_dynamic())
        {
            float alpha = 0.2;
            float beta  = 0.5;
            if(opd.onnx_name == "HardSwish")
            {
                alpha = 1.0 / 6.0;
            }
            else
            {
                if(contains(info.attributes, "alpha"))
                    alpha = info.attributes.at("alpha").f();

                if(contains(info.attributes, "beta"))
                    beta = info.attributes.at("beta").f();
            }

            auto input_lens = args[0]->get_shape().lens();
            auto input_type = args[0]->get_shape().type();

            // 插入shape算子
            std::vector<std::size_t> arg_shape = input_lens;
            std::vector<int64_t> vec_shape(arg_shape.size());
            migraphx::shape s(migraphx::shape::int64_type, {arg_shape.size()});
            std::transform(arg_shape.begin(), arg_shape.end(), vec_shape.begin(), [](auto i) {
                return int64_t(i);
            });
            instruction_ref shape_ins =
                info.add_instruction(op::Shape{migraphx::literal{s, vec_shape}}, {args[0]});

            auto mb_alpha = info.add_instruction(
                migraphx::make_op("multibroadcast_dynamic", {{"max_out_lens", input_lens}}),
                info.add_literal(migraphx::literal{migraphx::shape{input_type}, {alpha}}),
                shape_ins);
            auto mb_beta = info.add_instruction(
                migraphx::make_op("multibroadcast_dynamic", {{"max_out_lens", input_lens}}),
                info.add_literal(migraphx::literal{migraphx::shape{input_type}, {beta}}),
                shape_ins);
            auto mb_zero = info.add_instruction(
                migraphx::make_op("multibroadcast_dynamic", {{"max_out_lens", input_lens}}),
                info.add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}),
                shape_ins);
            auto mb_one = info.add_instruction(
                migraphx::make_op("multibroadcast_dynamic", {{"max_out_lens", input_lens}}),
                info.add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}),
                shape_ins);

            auto mul = info.add_instruction(migraphx::make_op("mul"), mb_alpha, args[0]);
            auto add = info.add_instruction(migraphx::make_op("add"), mb_beta, mul);
            auto hardsigmoid =
                info.add_instruction(migraphx::make_op("clip"), add, mb_zero, mb_one);
            if(opd.onnx_name == "HardSwish")
                return info.add_instruction(migraphx::make_op("mul"), args[0], hardsigmoid);

            return hardsigmoid;
        }
        else
        {
            float alpha = 0.2;
            float beta  = 0.5;
            if(opd.onnx_name == "HardSwish")
            {
                alpha = 1.0 / 6.0;
            }
            else
            {
                if(contains(info.attributes, "alpha"))
                    alpha = info.attributes.at("alpha").f();

                if(contains(info.attributes, "beta"))
                    beta = info.attributes.at("beta").f();
            }

            auto input_lens = args[0]->get_shape().lens();
            auto input_type = args[0]->get_shape().type();
            auto mb_alpha   = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                info.add_literal(migraphx::literal{migraphx::shape{input_type}, {alpha}}));
            auto mb_beta = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                info.add_literal(migraphx::literal{migraphx::shape{input_type}, {beta}}));
            auto mb_zero = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                info.add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
            auto mb_one = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                info.add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));

            auto mul = info.add_instruction(migraphx::make_op("mul"), mb_alpha, args[0]);
            auto add = info.add_instruction(migraphx::make_op("add"), mb_beta, mul);
            auto hardsigmoid =
                info.add_instruction(migraphx::make_op("clip"), add, mb_zero, mb_one);
            if(opd.onnx_name == "HardSwish")
                return info.add_instruction(migraphx::make_op("mul"), args[0], hardsigmoid);

            return hardsigmoid;
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
