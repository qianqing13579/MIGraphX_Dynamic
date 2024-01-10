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
#ifndef MIGRAPHX_GUARD_OPERATORS_RESHAPE_HPP
#define MIGRAPHX_GUARD_OPERATORS_RESHAPE_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <cmath>
#include <utility>
#include <migraphx/instruction.hpp> // 这个头文件不能删除，do_reshape中需要使用
#include <migraphx/instruction_ref.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reshape
{
    std::vector<int64_t> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

    value attributes() const { return {{"require_std_shape", true}}; }

    std::string name() const { return "reshape"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto&& idims = inputs.front().lens();
        std::vector<std::size_t> rdims(dims.begin(), dims.end());
        auto n_neg_dims = std::count(dims.begin(), dims.end(), -1);
        if(n_neg_dims > 1)
            MIGRAPHX_THROW("Reshape: Dimensions for reshape can only have one -1 dim");

        for(std::size_t i = 0; i < dims.size(); i++)
        {
            if(dims[i] == 0)
                rdims[i] = idims[i];

            // since rdims using size_t type, -1 is the max value
            // is size_t that cause later compuation incorrect
            if(dims[i] == -1)
                rdims[i] = 1;
        }

        if(n_neg_dims > 0)
        {
            size_t missing_dim =
                inputs.front().elements() /
                std::accumulate(rdims.begin(), rdims.end(), 1, std::multiplies<int64_t>());
            for(std::size_t i = 0; i < rdims.size(); i++)
            {
                if(dims[i] == -1)
                    rdims[i] = missing_dim;
            }
        }
        shape s{inputs.front().type(), rdims};
        if(s.elements() != inputs.front().elements())
            MIGRAPHX_THROW("Reshape: Wrong number of elements for reshape: reshape has " +
                           std::to_string(s.elements()) + " elements whereas the input has " +
                           std::to_string(inputs.front().elements()));
        return s;
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return args[0].reshape(output_shape);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }

    // dynamic shape
    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {

        // 返回-1出现的次数
        auto num_neg = std::count(this->dims.begin(), this->dims.end(), -1);

        // 获取输入输出属性
        std::vector<instruction_ref> inputs  = ins->inputs();
        std::vector<instruction_ref> outputs = ins->outputs();

        /*** 全局池化动态shape计算图模式：rewrite_pooling的第1个reshape
         *
         * reshape->reduce->reshape
         *
        resnet50计算图示例：
        main:@434 = reshape[dims={16384, -1}](main:@432) -> float_type, {16384, 49}, {49, 1}
        main:@435 = gpu::reduce_mean[axes={1}](main:@434,main:@433) -> float_type, {16384, 1}, {1,1} 
        main:@436 = load[offset=65536,end=97536](main:@1) -> float_type, {8, 1000}, {1000, 1}
        main:@437 = reshape[dims={8, 2048, 1, 1}](main:@435) -> float_type, {8, 2048, 1, 1}, {2048,1, 1, 1} main:@438 = flatten[axis=1](main:@437) -> float_type, {8, 2048}, {2048, 1}

        */
        if(num_neg == 1 &&
           ((ins->outputs()[0]->name() == "gpu::reduce_mean") ||
            (ins->outputs()[0]->name() == "gpu::reduce_max")) &&
           ins->outputs()[0]->outputs()[0]->name() == "reshape")
        {
            auto reduce_instruction = outputs[0];

            auto reshape_instruction = reduce_instruction->outputs()[0];
            std::vector<int64_t> dims_second_reshape =
                any_cast<migraphx::op::reshape>(reshape_instruction->get_operator()).dims;
            dims_second_reshape[0] = ins->inputs()[0]->get_shape().lens()[0];

            // 修改第一个reshape
            int count = std::accumulate(dims_second_reshape.begin(),
                                        dims_second_reshape.end(),
                                        1,
                                        std::multiplies<int64_t>());
            std::vector<int64_t> dims_first_reshape =
                any_cast<migraphx::op::reshape>(ins->get_operator()).dims;
            dims_first_reshape[0]          = count;
            int output_shape_first_reshape = ins->inputs()[0]->get_shape().elements() / count;

            // 修改reduce算子的输出，注意：此时不能直接替换op，需要使用set_output_shape()先修改输出shape
            std::vector<std::size_t> lens(2, 1);
            lens[0] = count;
            lens[1] = 1;
            reduce_instruction->set_output_shape(
                shape(reduce_instruction->get_shape().type(), lens));

            // 修改第二个reshape
            any_cast<migraphx::op::reshape>(reshape_instruction->get_operator()).dims =
                dims_second_reshape;
            shape new_shape = migraphx::compute_shape(reshape_instruction->get_operator(),
                                                      reshape_instruction->inputs(),
                                                      reshape_instruction->module_inputs());
            reshape_instruction->set_output_shape(new_shape);

            // 修改第一个reshape
            lens[1]    = output_shape_first_reshape;
            this->dims = dims_first_reshape;
        }

        return true;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
