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
#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_GEMM_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_GEMM_HPP

#include <migraphx/errors.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/value.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/gemm_impl.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

void blas_shape(const shape& s);
shape transpose_batch(const shape& s, unsigned trans_batch);

template <class Op>
struct rocblas_gemm
{
    Op op;
    float alpha          = 1;
    float beta           = 0;
    bool int8_x4_format  = true;
    bool compute_fp32    = false;
    unsigned trans_batch = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack_join(migraphx::reflect(self.op, f),
                         pack(f(self.alpha, "alpha"),
                              f(self.beta, "beta"),
                              f(self.int8_x4_format, "int8_x4_format"),
                              f(self.compute_fp32, "compute_fp32"),
                              f(self.trans_batch, "trans_batch")));
    }

    std::string name() const
    {
        if(contains(op.name(), "quant_"))
        {
            return "gpu::quant_gemm";
        }
        return "gpu::gemm";
    }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        std::vector<shape> in_shapes(inputs);
        in_shapes.pop_back();
        check_shapes{in_shapes, *this}.not_broadcasted();
        blas_shape(inputs[0]);
        blas_shape(inputs[1]);
        // if gemm and add are fused
        if(in_shapes.size() > 2)
        {
            auto cmat_shape = in_shapes.back();
            check_shapes{{cmat_shape}, *this}.not_transposed().not_broadcasted();
            in_shapes.pop_back();
            blas_shape(cmat_shape);
            auto op_out_shape = op.compute_shape(in_shapes);
            if(cmat_shape.lens() != op_out_shape.lens())
            {
                MIGRAPHX_THROW(this->name() + " : dimension mismatch, operand C: {" +
                               to_string_range(cmat_shape.lens()) +
                               "}, cannot add to operand A * B: {" +
                               to_string_range(op_out_shape.lens()) + "}");
            }
            if(cmat_shape.type() != op_out_shape.type())
            {
                MIGRAPHX_THROW(this->name() + " : operand C type mismatch, operand C is of type: " +
                               to_string(cmat_shape.type()) +
                               ", it must be: " + to_string(op_out_shape.type()));
            }
            return transpose_batch(op_out_shape, trans_batch);
        }

        return transpose_batch(op.compute_shape(in_shapes), trans_batch);
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        if(this->name() == "gpu::gemm")
        {
            gemm(ctx, output_shape, args, alpha, beta, int8_x4_format, compute_fp32);
        }
        else
        {
            gemm(ctx,
                 output_shape,
                 args,
                 int32_t(alpha),
                 int32_t(beta),
                 int8_x4_format,
                 compute_fp32);
        }
        return args.back();
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    bool do_reshape(instruction_ref ins,std::unordered_map<instruction_ref, argument> &results)
    {
        instruction_ref input0=ins->inputs()[0];
        shape shape_input0=input0->get_shape();

        /* 模式1：处理维度>2的gemm，需要修改gemm算子第2个参数的shape，第2个参数最后两个维度是固定的，前面的维度都是广播产生
        
        计算图示例：gpt2-lm-head-10_NxNxN.onnx
        // 未编译计算图
        main:@3664 = reshape_dynamic[max_dims={2, 32, 64, 768}](main:@3658,main:@3663) -> float_type, {2, 32, 64, 768}, {1572864, 49152, 768, 1}
        main:@3665 = multibroadcast[out_lens={2, 32, 768, 50257}](main:@819) -> float_type, {2, 32, 768, 50257}, {0, 0, 50257, 1}
        main:@3666 = dot(main:@3664,main:@3665) -> float_type, {2, 32, 64, 50257}, {102926336, 3216448, 50257, 1}
        // 编译后计算图
        main:@3765 = gpu::reshape_dynamic[max_dims={2, 32, 64, 768}](main:@3749,main:@3764) -> float_type, {2, 32, 64, 768}, {1572864, 49152, 768, 1}
        main:@3766 = hip::hip_copy_literal[id=main:@literal:0] -> float_type, {2, 32, 768, 50257}, {1235116032, 38597376, 50257, 1}
        main:@3767 = load[offset=289406992,end=1112817680](main:@1) -> float_type, {2, 32, 64, 50257}, {102926336, 3216448, 50257, 1}
        main:@3768 = gpu::gemm[alpha=1,beta=0,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@3765,main:@3766,main:@3767) -> float_type, {2, 32, 64, 50257}, {102926336, 3216448, 50257, 1}
        main:@3769 = load[offset=138412048,end=163577872](main:@1) -> float_type, {2, 64, 12, 64, 64}, {3145728, 49152, 4096, 64, 1}
        */
        if(shape_input0.lens().size()>2&&ins->inputs()[1]->name()=="hip::hip_copy_literal")
        {
            instruction_ref input1=ins->inputs()[1];
            shape old_shape=input1->get_shape();
            std::vector<std::size_t> new_lens=old_shape.lens();
            for(int i=0;i<shape_input0.lens().size()-2;++i)
            {
                new_lens[i]=shape_input0.lens()[i];
            }
            shape new_shape{old_shape.type(),new_lens};
            any_cast<migraphx::gpu::hip_copy_literal>(input1->get_operator()).reshape_param(new_shape);
            input1->set_output_shape(new_shape);
        }

        // 如果融合了add算子,还需要修改第3个参数，这里的if和上面的if不是互斥关系，上面的执行完了，再执行这里的if判断
        std::vector<instruction_ref> inputs = ins->inputs();
        if(inputs.size() > 3)
        {
            shape old_shape2=inputs[2]->get_shape();
            std::vector<std::size_t> new_lens2=old_shape2.lens();
            for(int i=0;i<new_lens2.size()-1;++i)
            {
                new_lens2[i]=shape_input0.lens()[i];
            }
            
            shape new_shape2{old_shape2.type(),new_lens2};

            /* 第3个输入参数是hip::copy,而hip::copy算子的输入是hip::hip_copy_literal
            
            注意：目前代码中保留对hip::copy情况的处理，是为了兼容官方以前find_gemm_add的实现，官方之前在find_gemm_add的实现中添加了hip::copy指令，导致了融合后的gemm算子输出参数也是hip::copy指令。
            参考Commit: 99b4811b6cbcc2b17364681f4143820008e2f470
            */
            if(inputs[2]->get_operator().name() == "hip::copy")
            {
                // 修改hip::copy的输入参数hip_copy_literal算子
                instruction_ref hip_copy_literal=inputs[2]->inputs()[0];
                any_cast<migraphx::gpu::hip_copy_literal>(hip_copy_literal->get_operator()).reshape_param(new_shape2);
                hip_copy_literal->set_output_shape(new_shape2);

                // 对hip::copy算子进行reshape
                inputs[2]->reshape(inputs[2],results);
            }

            // 第3个输入参数是hip::hip_copy_literal
            if(inputs[2]->get_operator().name() == "hip::hip_copy_literal")
            {
                instruction_ref hip_copy_literal=inputs[2];
                any_cast<migraphx::gpu::hip_copy_literal>(hip_copy_literal->get_operator()).reshape_param(new_shape2);
                hip_copy_literal->set_output_shape(new_shape2);
            }

        }
        

        return true;
        
    }

};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
