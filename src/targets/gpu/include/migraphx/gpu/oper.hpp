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
#ifndef MIGRAPHX_GUARD_RTGLIB_UNARY_HPP
#define MIGRAPHX_GUARD_RTGLIB_UNARY_HPP

#include <migraphx/gpu/name.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/type_name.hpp>
#include <utility>
#include <iostream>
#include <migraphx/common.hpp>
#include <migraphx/op/multibroadcast.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

template <class Derived, std::size_t N>
struct device_base : oper<Derived>
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    value attributes() const { return {{"pointwise", true}}; }

    std::vector<shape> reduce_shapes;

    void finalize(context&, const shape&, const std::vector<shape>& inputs)
    {
        reduce_shapes = reduce_dims(inputs);
    }

    argument get_arg(const std::vector<argument>& args, std::size_t i) const
    {
        if(reduce_shapes.empty())
            return args[i];
        return args.at(i).reshape(reduce_shapes.at(i));
    }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(N + 1);
        auto s0 = inputs.at(0);
        if(std::all_of(inputs.begin(), inputs.end() - 1, [&](auto s) { return s == s0; }) and
           s0.packed())
            return s0;
        else
            return {s0.type(), s0.lens()};
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    // dynamic shape
    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        /**
         *  如果操作数都是广播，则重新计算广播后的大小
         *
         * 示例计算图：
        main:@1927 = gpu::reshape_dynamic[max_dims={-1, 1, 1, 1}](main:@1912,main:@68) ->float_type, {2, 1, 1, 1}, {1, 1, 1, 1} 
        main:@1928 = gpu::reshape_dynamic[max_dims={-1, 1,1}](main:@1917,main:@79) -> float_type, {11, 1, 1}, {1, 1, 1} 
        main:@1931 = multibroadcast[out_lens={2, 11, 1, 1}](main:@1927) -> float_type, {2, 11, 1, 1}, {1, 0, 1,1} 
        main:@1932 = multibroadcast[out_lens={2, 11, 1, 1}](main:@1928) -> float_type, {2, 11, 1,1}, {0, 1, 1, 1} 
        main:@1933 = load[offset=5376,end=5464](main:@1) -> float_type, {2, 11, 1,1}, {11, 1, 1, 1} 
        main:@1934 = gpu::add(main:@1932,main:@1931,main:@1933) -> float_type, {2,11, 1, 1}, {11, 1, 1, 1}
        */
        if(std::all_of(ins->inputs().begin(), ins->inputs().end() - 1, [](auto ins) {
               return ins->name() == "multibroadcast";
           }))
        {
            // 获取所有multibroadcast指令的输入shape
            std::vector<shape> input_shapes;
            for(int i = 0; i <= ins->inputs().size() - 2; ++i)
            {
                instruction_ref multibroadcast_ins = ins->inputs()[i];
                input_shapes.push_back(multibroadcast_ins->inputs()[0]->get_shape());
            }

            // 重新计算广播后的大小
            shape output_shape = common_shape(input_shapes);

            // 修改属性，并重新计算输出大小
            for(int i = 0; i <= ins->inputs().size() - 2; ++i)
            {
                instruction_ref multibroadcast_ins = ins->inputs()[i];
                any_cast<op::multibroadcast>(multibroadcast_ins->get_operator()).output_lens =
                    output_shape.lens();

                shape new_shape = migraphx::compute_shape(multibroadcast_ins->get_operator(),
                                                          multibroadcast_ins->inputs(),
                                                          multibroadcast_ins->module_inputs());
                multibroadcast_ins->set_output_shape(new_shape);

                // 修改输入tensor
                results[multibroadcast_ins] =
                    results[multibroadcast_ins].reshape(multibroadcast_ins->get_shape());
            }
        }
        /**  
         * 如果有操作数是常量，则取view
         * 
        GatherElements算子:
        main:@89 = hip::hip_copy_literal[id=main:@literal:87] -> float_type, {4, 11, 64}, {704, 64, 1}
        main:@93 = hip::hip_copy_literal[id=main:@literal:91] -> float_type, {4, 11, 64}, {704, 64, 1}
        main:@77 = hip::hip_copy_literal[id=main:@literal:75] -> float_type, {4, 11, 64}, {704, 64, 1}
        main:@445 = gpu::multibroadcast_dynamic[max_out_lens={4, 11, 64}](main:@444,main:@442) -> float_type, {4, 11, 64}, {11, 1, 0}
        main:@448 = gpu::mul(main:@408,main:@446,main:@447) -> float_type, {4, 23, 64}, {1472, 64, 1}
        main:@449 = reshape[dims={5888}](main:@448) -> float_type, {5888}, {1}
        main:@450 = load[offset=1072,end=12336](main:@1) -> float_type, {4, 11, 64}, {704, 64, 1}
        main:@451 = gpu::sub(main:@445,main:@89,main:@450) -> float_type, {4, 11, 64}, {704, 64, 1}
        main:@452 = load[offset=43012,end=54276](main:@1) -> float_type, {4, 11, 64}, {704, 64, 1}
        main:@453 = gpu::mul_add(main:@451,main:@93,main:@77,main:@452) -> float_type, {4, 11, 64}, {704, 64, 1}
        main:@454 = load[offset=8196,end=19460](main:@1) -> float_type, {4, 11, 64}, {704, 64, 1}
        main:@455 = gpu::gather[axis=0](main:@449,main:@453,main:@454) -> float_type, {4, 11, 64}, {704, 64, 1}
         * 
        */
        else if(std::any_of(ins->inputs().begin() + 1, ins->inputs().end() - 1, [](auto ins) {
                    return ins->name() == "hip::hip_copy_literal";
                }))
        {

            // 记录非常量操作数
            std::vector<std::size_t> output_lens;
            for(int i = 0; i <= ins->inputs().size() - 2; ++i)
            {
                if(ins->inputs()[i]->name() != "hip::hip_copy_literal")
                {
                    output_lens = ins->inputs()[i]->get_shape().lens();
                    break;
                }
            }

            // 对于常量的操作数，取view
            for(int i = 0; i <= ins->inputs().size() - 2; ++i)
            {
                if(ins->inputs()[i]->name() == "hip::hip_copy_literal")
                {
                    instruction_ref hip_copy_literal = ins->inputs()[i];
                    shape new_shape{hip_copy_literal->get_shape().type(),
                                    output_lens,
                                    hip_copy_literal->get_shape().strides()};
                    any_cast<migraphx::gpu::hip_copy_literal>(hip_copy_literal->get_operator())
                        .reshape_param(new_shape);
                    hip_copy_literal->set_output_shape(new_shape);

                    // 修改输入tensor
                    results[hip_copy_literal] =
                        results[hip_copy_literal].reshape(hip_copy_literal->get_shape());
                }
            }
        }
        /**
         * 如果有操作数是"multibroadcast+非标量常量"，取view
         *
         * 由于目前MIGraphX保留了常量传播，所以有可能出现这种情况，比如以前的yolov5和gpt2:
         *
        GPT2:
        main:@63 = hip::hip_copy_literal[id=main:@literal:38] -> bool_type, {1, 1, 64, 64}, {4096,4096, 64, 1} 
        main:@69 = gpu::mul(main:@61,main:@67,main:@68) -> float_type, {1, 12, 64, 64},{49152, 4096, 64, 1} 
        main:@72 = multibroadcast[out_lens={1, 12, 64, 64}](main:@63) -> bool_type, {1, 12, 64, 64}, {4096, 0, 64, 1} 
        main:@73 = gpu::where(main:@72,main:@69,main:@71,main:@70) -> float_type, {1, 12, 64, 64}, {49152,4096, 64, 1}

        YOLOV5:
        main:@885 = hip::hip_copy_literal[id=main:@literal:2] -> float_type, {1, 3, 40, 40, 2},{9600, 3200, 80, 2, 1} 
        main:@898 = multibroadcast[out_lens={2, 3, 40, 40, 2}](main:@885) -> float_type, {2, 3, 40, 40, 2}, {0, 3200, 80, 2, 1} 
        main:@899 = gpu::mul_add_mul(main:@897,main:@891,main:@898,main:@896,main:@895) -> float_type, {2, 3,40, 40, 2}, {9600, 3200, 80, 2, 1}
         *
        */
        else if(std::any_of(ins->inputs().begin(), ins->inputs().end() - 1, [](auto ins) {
                    return ins->name() == "multibroadcast" &&
                           ins->inputs()[0]->name() == "hip::hip_copy_literal" &&
                           ins->inputs()[0]->get_shape().scalar() == false;
                }))
        {
            // 如果有操作数是"multibroadcast+非标量常量"，取view
            for(int i = 0; i <= ins->inputs().size() - 2; ++i)
            {
                if(ins->inputs()[i]->name() == "multibroadcast" &&
                   ins->inputs()[i]->inputs()[0]->name() == "hip::hip_copy_literal" &&
                   ins->inputs()[i]->inputs()[0]->get_shape().scalar() == false)
                {
                    instruction_ref multibroadcast_ins   = ins->inputs()[i];
                    instruction_ref hip_copy_literal_ins = multibroadcast_ins->inputs()[0];

                    std::vector<std::size_t> old_lens = hip_copy_literal_ins->get_shape().lens();
                    std::vector<std::size_t> output_lens =
                        any_cast<op::multibroadcast>(multibroadcast_ins->get_operator())
                            .output_lens;
                    int offset = output_lens.size() - old_lens.size();
                    for(int i = old_lens.size() - 1; i >= 0; i--)
                    {
                        if(output_lens[i + offset] != old_lens[i] and old_lens[i] != 1)
                        {
                            // 表示这是不需要广播的维度，应该保持不变
                            old_lens[i] = output_lens[i + offset];
                        }
                    }

                    // 修改常量的shape,创建原来常量的一个view:注意只需要改变lens,步长和原来保持一致
                    shape new_shape{hip_copy_literal_ins->get_shape().type(),
                                    old_lens,
                                    hip_copy_literal_ins->get_shape().strides()};
                    any_cast<migraphx::gpu::hip_copy_literal>(hip_copy_literal_ins->get_operator())
                        .reshape_param(new_shape);
                    hip_copy_literal_ins->set_output_shape(new_shape);
                    results[hip_copy_literal_ins] =
                        results[hip_copy_literal_ins].reshape(hip_copy_literal_ins->get_shape());

                    // 重新计算广播算子
                    shape new_shape2 = migraphx::compute_shape(multibroadcast_ins->get_operator(),
                                                               multibroadcast_ins->inputs(),
                                                               multibroadcast_ins->module_inputs());
                    multibroadcast_ins->set_output_shape(new_shape2);
                    results[multibroadcast_ins] =
                        results[multibroadcast_ins].reshape(multibroadcast_ins->get_shape());
                }
            }
        }
        return true;
    }
};

template <class Derived, void (*F)(hipStream_t, const argument&, const argument&)>
struct unary_device : device_base<Derived, 1>
{
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(), this->get_arg(args, 1), this->get_arg(args, 0));
        return args[1];
    }
};

template <class Derived, void (*F)(hipStream_t, const argument&, const argument&, const argument&)>
struct binary_device : device_base<Derived, 2>
{
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(),
          this->get_arg(args, 2),
          this->get_arg(args, 0),
          this->get_arg(args, 1));
        return args[2];
    }
};

template <class Derived,
          void (*F)(
              hipStream_t, const argument&, const argument&, const argument&, const argument&)>
struct ternary_device : device_base<Derived, 3>
{
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(),
          this->get_arg(args, 3),
          this->get_arg(args, 0),
          this->get_arg(args, 1),
          this->get_arg(args, 2));
        return args[3];
    }
};

template <class Derived,
          void (*F)(hipStream_t,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&)>
struct quaternary_device : device_base<Derived, 4>
{
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(),
          this->get_arg(args, 4),
          this->get_arg(args, 0),
          this->get_arg(args, 1),
          this->get_arg(args, 2),
          this->get_arg(args, 3));
        return args[4];
    }
};

template <class Derived,
          void (*F)(hipStream_t,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&)>
struct quinary_device : device_base<Derived, 5>
{
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(),
          this->get_arg(args, 5),
          this->get_arg(args, 0),
          this->get_arg(args, 1),
          this->get_arg(args, 2),
          this->get_arg(args, 3),
          this->get_arg(args, 4));
        return args[5];
    }
};

template <class Derived,
          void (*F)(hipStream_t,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&)>
struct six_device : device_base<Derived, 6>
{
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(),
          this->get_arg(args, 6),
          this->get_arg(args, 0),
          this->get_arg(args, 1),
          this->get_arg(args, 2),
          this->get_arg(args, 3),
          this->get_arg(args, 4),
          this->get_arg(args, 5));
        return args[6];
    }
};

template <class Derived,
          void (*F)(hipStream_t,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&,
                    const argument&)>
struct eight_device : device_base<Derived, 8>
{
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        F(ctx.get_stream().get(),
          this->get_arg(args, 8),
          this->get_arg(args, 0),
          this->get_arg(args, 1),
          this->get_arg(args, 2),
          this->get_arg(args, 3),
          this->get_arg(args, 4),
          this->get_arg(args, 5),
          this->get_arg(args, 6),
          this->get_arg(args, 7));
        return args[8];
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
