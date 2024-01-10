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
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_scatter_elements : op_parser<parse_scatter_elements>
{
    std::vector<op_desc> operators() const { return {{"ScatterElements"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        int axis      = 0;
        int reduction = 0;

        // 限制条件
        auto data_shape    = args[0]->get_shape();
        auto indices_shape = args[1]->get_shape();
        auto updates_shape = args[2]->get_shape();

        if(data_shape.type() != updates_shape.type())
        {
            MIGRAPHX_THROW("PARSE_SCATTER_ELEMENTS: data type is different from updates type");
        }

        // 保证indices和updates具有相同的维度
        if(indices_shape.lens().size() != updates_shape.lens().size())
        {
            MIGRAPHX_THROW("PARSE_SCATTER_ELEMENTS: Indices and updates must have the same rank");
        }

        // 保证在各个维度上具有相同的数量
        for(size_t i = 0; i < indices_shape.lens().size(); ++i)
        {
            if(indices_shape.lens()[i] != updates_shape.lens()[i])
            {
                MIGRAPHX_THROW("PARSE_SCATTER_ELEMENTS: indices and updates must have same numbers "
                               "of dimensions");
            }
        }

        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        if(contains(info.attributes, "reduction"))
        {
            std::string reduction_att(info.attributes.at("reduction").s());

            if(reduction_att == "none")
            {
                reduction = 0;
            }
            else if(reduction_att == "add")
            {
                reduction = 1;
            }
            else if(reduction_att == "mul")
            {
                reduction = 2;
            }
            else if(reduction_att == "min")
            {
                reduction = 3;
            }
            else
            {
                reduction = 4;
            }
        }
        return info.add_instruction(
            migraphx::make_op("scatter_elements", {{"axis", axis}, {"reduction", reduction}}),
            args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx