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
#ifndef MIGRAPHX_GUARD_OPERATORS_MULTIBROADCAST_HPP
#define MIGRAPHX_GUARD_OPERATORS_MULTIBROADCAST_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/lifetime.hpp>
#include <migraphx/instruction.hpp> // 这个头文件不能删除，do_reshape中需要使用
#include <migraphx/instruction_ref.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct multibroadcast
{
    std::vector<std::size_t> output_lens;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_lens, "out_lens"));
    }

    std::string name() const { return "multibroadcast"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto t     = inputs.at(0).type();
        auto input = inputs.at(0);

        if(input.lens().empty())
        {
            MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should be > 0");
        }

        if(input.lens().size() > output_lens.size())
        {
            MIGRAPHX_THROW("MULTIBROADCAST: inputs dimensions should <= output size");
        }

        auto offset = output_lens.size() - input.lens().size();
        for(std::ptrdiff_t i = input.lens().size() - 1; i >= 0; i--)
        {
            if(output_lens[i + offset] != input.lens()[i] and input.lens()[i] != 1)
            {
                MIGRAPHX_THROW("MULTIBROADCAST: input shape {" + to_string_range(input.lens()) +
                               "} cannot be broadcasted to {" + to_string_range(output_lens) +
                               "}!");
            }
        }

        std::vector<size_t> bcast_strides(output_lens.size(), 0);
        for(std::ptrdiff_t i = input.lens().size() - 1; i >= 0; i--)
        {
            if(output_lens[i + offset] == input.lens()[i])
            {
                bcast_strides[i + offset] = input.strides()[i];
            }
        }
        return {t, output_lens, bcast_strides};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return args[0].reshape(output_shape);
    }
    lifetime get_lifetime() const { return lifetime::borrow; }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }

    // dynamic shape
    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        //////////////////////////////////////////////////////
        ///先使用通用动态shape模式匹配器修改算子属性 ////////////////////////////////////
        // 寻找broadcast的输出节点的另一个输入节点
        auto output = ins->outputs()[0];

        /**
         *  如果输出指令是pointwise且所有输入参数都是广播，则此时不需要重新计算shape
         *
         * 示例计算图：远鉴模型 ModelZoo/Others/YuanJian/model_dynamic_len.onnx
        main:@1927 = gpu::reshape_dynamic[max_dims={-1, 1, 1, 1}](main:@1912,main:@68) ->float_type, {2, 1, 1, 1}, {1, 1, 1, 1} 
        main:@1928 = gpu::reshape_dynamic[max_dims={-1, 1,1}](main:@1917,main:@79) -> float_type, {11, 1, 1}, {1, 1, 1} 
        main:@1931 = multibroadcast[out_lens={2, 11, 1, 1}](main:@1927) -> float_type, {2, 11, 1, 1}, {1, 0, 1,1} 
        main:@1932 = multibroadcast[out_lens={2, 11, 1, 1}](main:@1928) -> float_type, {2, 11, 1,1}, {0, 1, 1, 1} 
        main:@1933 = load[offset=5376,end=5464](main:@1) -> float_type, {2, 11, 1,1}, {11, 1, 1, 1} 
        main:@1934 = gpu::add(main:@1932,main:@1931,main:@1933) -> float_type, {2,11, 1, 1}, {11, 1, 1, 1}
        */
        if(output->get_operator().attributes().contains("pointwise") &&
           std::all_of(output->inputs().begin(), output->inputs().end() - 1, [](auto ins) {
               return ins->name() == "multibroadcast";
           }))
        {
            return false;
        }
        // 如果broadcast的输出节点是contiguous或者convert
        else if(output->name() == "gpu::contiguous" || output->name() == "gpu::convert")
        {
            // 继续寻找contiguous的输出节点
            auto output2 = output->outputs()[0];

            // 采用另一个输入的大小
            auto inputsOfOutput = output2->inputs();
            for(auto input : inputsOfOutput)
            {
                // 使用输出节点的另一个输入大小
                if((input->get_shape().lens() != output->get_shape().lens()) &&
                   (input->name() != "load"))
                {
                    // 修改算子属性
                    this->output_lens = input->get_shape().lens();
                    break;
                }
            }
        }
        else
        {
            // 采用另一个输入的大小
            auto inputsOfOutput = output->inputs();
            for(auto input : inputsOfOutput)
            {
                // 使用输出节点的另一个输入大小
                if((input->get_shape().lens() != ins->get_shape().lens()) &&
                   (input->name() != "load"))
                {
                    // 修改算子属性
                    this->output_lens = input->get_shape().lens();
                    break;
                }
            }
        }

        /**
         * "multibroadcast+非标量常量",则返回false,不需要计算shape
         *
        GPT2:
        main:@63 = hip::hip_copy_literal[id=main:@literal:38] -> bool_type, {1, 1, 64, 64}, {4096,4096, 64, 1} 
        main:@69 = gpu::mul(main:@61,main:@67,main:@68) -> float_type, {1, 12, 64, 64},{49152, 4096, 64, 1} 
        main:@72 = multibroadcast[out_lens={1, 12, 64, 64}](main:@63) ->bool_type, {1, 12, 64, 64}, {4096, 0, 64, 1} 
        main:@73 = gpu::where(main:@72,main:@69,main:@71,main:@70) -> float_type, {1, 12, 64, 64}, {49152,4096, 64, 1}

        YOLOV5:
        main:@885 = hip::hip_copy_literal[id=main:@literal:2] -> float_type, {1, 3, 40, 40, 2},{9600, 3200, 80, 2, 1} 
        main:@898 = multibroadcast[out_lens={2, 3, 40, 40, 2}](main:@885) ->float_type, {2, 3, 40, 40, 2}, {0, 3200, 80, 2, 1} 
        main:@899 = gpu::mul_add_mul(main:@897,main:@891,main:@898,main:@896,main:@895) -> float_type, {2, 3,40, 40, 2}, {9600, 3200, 80, 2, 1}
         *
         *
        */
        if(output->get_operator().attributes().contains("pointwise") &&
           ins->inputs()[0]->name() == "hip::hip_copy_literal" &&
           ins->inputs()[0]->get_shape().scalar() == false)
        {
            return false;
        }

        return true;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
