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
    bool do_reshape(instruction_ref ins,std::unordered_map<instruction_ref, argument> &results)
    {

        // 返回-1出现的次数
        auto num_neg = std::count(this->dims.begin(), this->dims.end(), -1);

        // 获取输入输出属性
        std::vector<instruction_ref> inputs=ins->inputs();
        std::vector<instruction_ref> outputs=ins->outputs();

        // 是否存在lstm算子
        instruction_ref lstm_ins=ins;
        bool hasLSTM=false;
        while(lstm_ins->inputs().size()>0)
        {
            if(lstm_ins->inputs()[0]->name()=="gpu::lstm")
            {
                lstm_ins=lstm_ins->inputs()[0];
                hasLSTM=true;
                break;
            }
            lstm_ins=lstm_ins->inputs()[0];
        }

        
        /*** 分类模型中的全局池化动态shape计算图模式：rewrite_pooling的第1个reshape
         * 
         * reshape->reduce->reshape
         * 
        resnet50计算图示例：
        main:@434 = reshape[dims={16384, -1}](main:@432) -> float_type, {16384, 49}, {49, 1}
        main:@435 = gpu::reduce_mean[axes={1}](main:@434,main:@433) -> float_type, {16384, 1}, {1, 1}
        main:@436 = load[offset=65536,end=97536](main:@1) -> float_type, {8, 1000}, {1000, 1}
        main:@437 = reshape[dims={8, 2048, 1, 1}](main:@435) -> float_type, {8, 2048, 1, 1}, {2048, 1, 1, 1}
        main:@438 = flatten[axis=1](main:@437) -> float_type, {8, 2048}, {2048, 1}

        */
        if(
           num_neg==1&&
           ((ins->outputs()[0]->name()=="gpu::reduce_mean")||
            (ins->outputs()[0]->name()=="gpu::reduce_max"))&&
            ins->outputs()[0]->outputs()[0]->name()=="reshape"
          )
        {
            auto reduce_instruction=outputs[0];

            auto reshape_instruction=reduce_instruction->outputs()[0];
            std::vector<int64_t> dims_second_reshape= any_cast<migraphx::op::reshape>(reshape_instruction->get_operator()).dims;
            dims_second_reshape[0]=ins->inputs()[0]->get_shape().lens()[0];

            // 修改第一个reshape
            int count=std::accumulate(dims_second_reshape.begin(), dims_second_reshape.end(), 1, std::multiplies<int64_t>());
            std::vector<int64_t> dims_first_reshape= any_cast<migraphx::op::reshape>(ins->get_operator()).dims;
            dims_first_reshape[0]=count;
            int output_shape_first_reshape=ins->inputs()[0]->get_shape().elements()/count;
            
            // 修改reduce算子的输出，注意：此时不能直接替换op，需要使用set_output_shape()先修改输出shape
            std::vector<std::size_t> lens(2,1);
            lens[0]=count;
            lens[1]=1;
            reduce_instruction->set_output_shape(shape(reduce_instruction->get_shape().type(),lens));

            // 修改第二个reshape
            any_cast<migraphx::op::reshape>(reshape_instruction->get_operator()).dims=dims_second_reshape;
            shape new_shape=migraphx::compute_shape(reshape_instruction->get_operator(),reshape_instruction->inputs() , reshape_instruction->module_inputs());
            reshape_instruction->set_output_shape(new_shape);

            // 修改第一个reshape
            lens[1]=output_shape_first_reshape;
            this->dims=dims_first_reshape;

        }
        /*** 分类模型中的全局池化动态shape计算图模式：rewrite_pooling的第2个reshape
         * 
         * reshape->reduce->reshape
         * 
        resnet50计算图示例：
        main:@434 = reshape[dims={16384, -1}](main:@432) -> float_type, {16384, 49}, {49, 1}
        main:@435 = gpu::reduce_mean[axes={1}](main:@434,main:@433) -> float_type, {16384, 1}, {1, 1}
        main:@436 = load[offset=65536,end=97536](main:@1) -> float_type, {8, 1000}, {1000, 1}
        main:@437 = reshape[dims={8, 2048, 1, 1}](main:@435) -> float_type, {8, 2048, 1, 1}, {2048, 1, 1, 1}
        main:@438 = flatten[axis=1](main:@437) -> float_type, {8, 2048}, {2048, 1}

        */
        else if(
                num_neg==0&&
                ((ins->inputs()[0]->name()=="gpu::reduce_mean")||
                (ins->inputs()[0]->name()=="gpu::reduce_max"))&&
                ins->inputs()[0]->inputs()[0]->name()=="reshape"
                )
        {
            // 不处理
        }
        /*** 分类模型中的全局池化动态shape计算图模式
         * 
         * GlobalAveragePool->Reshape，此时的reshape的作用相当于flatten,只修改batchsize
         * 
         mobilenet_v2计算图示例：
         main:@450 = reshape[dims={1280, -1}](main:@448) -> float_type, {1280, 49}, {49, 1}
         main:@451 = gpu::reduce_mean[axes={1}](main:@450,main:@449) -> float_type, {1280, 1}, {1, 1}
         main:@452 = load[offset=0,end=4000](main:@1) -> float_type, {1, 1000}, {1000, 1}
         main:@453 = reshape[dims={1, 1280, 1, 1}](main:@451) -> float_type, {1, 1280, 1, 1}, {1280, 1, 1, 1}
         main:@454 = reshape[dims={1, -1}](main:@453) -> float_type, {1, 1280}, {1280, 1}
        
         paddleocr_cls计算图示例：
         main:@738 = reshape[dims={800, -1}](main:@736) -> float_type, {800, 50}, {50, 1}
         main:@739 = gpu::reduce_mean[axes={1}](main:@738,main:@737) -> float_type, {800, 1}, {1, 1}
         main:@740 = reshape[dims={4, 200, 1, 1}](main:@739) -> float_type, {4, 200, 1, 1}, {200, 1, 1, 1}
         main:@741 = reshape[dims={4, 200}](main:@740) -> float_type, {4, 200}, {200, 1}
         * 
        */
        else if(
                ins->inputs()[0]->name()=="reshape"&&
                ((ins->inputs()[0]->inputs()[0]->name()=="gpu::reduce_mean")||
                (ins->inputs()[0]->inputs()[0]->name()=="gpu::reduce_max"))&&
                ins->inputs()[0]->inputs()[0]->inputs()[0]->name()=="reshape"&&
                dims.size()==2
                )
        {
            dims[0]=ins->inputs()[0]->get_shape().lens()[0];
            dims[1]=-1;
        }
        /**
         *  GatherElements算子
         * 
         * 示例计算图：
        main:@448 = gpu::mul(main:@408,main:@446,main:@447) -> float_type, {2, 23, 64}, {1472, 64, 1}
        main:@449 = reshape[dims={2944}](main:@448) -> float_type, {2944}, {1}
        main:@450 = load[offset=536,end=6168](main:@1) -> float_type, {2, 11, 64}, {704, 64, 1}
        main:@451 = gpu::sub(main:@445,main:@82,main:@450) -> float_type, {2, 11, 64}, {704, 64, 1}
        main:@452 = load[offset=21508,end=27140](main:@1) -> float_type, {2, 11, 64}, {704, 64, 1}
        main:@453 = gpu::mul_add(main:@451,main:@89,main:@76,main:@452) -> float_type, {2, 11, 64}, {704, 64, 1}
        main:@454 = load[offset=4100,end=9732](main:@1) -> float_type, {2, 11, 64}, {704, 64, 1}
        main:@455 = gpu::gather[axis=0](main:@449,main:@453,main:@454) -> float_type, {2, 11, 64}, {704, 64, 1}
         */
        else if(ins->outputs()[0]->name()=="gpu::gather"&&
                dims.size()==1&&
                ins->outputs()[0]->inputs()[1]->name()=="gpu::mul_add"&&
                ins->outputs()[0]->inputs()[1]->inputs()[0]->name()=="gpu::sub")
        {
            dims[0]=ins->inputs()[0]->get_shape().elements();
        }
        /*** RetinaFace模型中的动态shape计算图模式
         * 
         * Conv->Transpose->Reshape->Concat
         * 
         计算图示例：
         main:@610 = gpu::convolution[padding={0, 0, 0, 0},stride={1, 1},dilation={1, 1},group=1,padding_mode=0,solution_id=128](main:@598,main:@607,main:@609,main:@608) -> float_type, {1, 8, 10, 10}, {800, 100, 10, 1}
         main:@686 = gpu::add(main:@610,main:@684,main:@685) -> float_type, {1, 8, 10, 10}, {800, 100, 10, 1}
         main:@687 = transpose[permutation={0, 2, 3, 1}](main:@686) -> float_type, {1, 10, 10, 8}, {800, 10, 1, 100}
         main:@689 = gpu::contiguous(main:@687,main:@688) -> float_type, {1, 10, 10, 8}, {800, 80, 8, 1}
         main:@747 = reshape[dims={1, -1, 4}](main:@689) -> float_type, {1, 200, 4}, {800, 4, 1}
         main:@750 = gpu::concat[axis=1](main:@749,main:@748,main:@747,main:@746) -> float_type, {1, 4200, 4}, {16800, 4, 1}
         * 
        */
        else if(
                ins->inputs()[0]->name()=="gpu::contiguous"&& // 由于reshape不支持view，所以需要添加contiguous
                ins->inputs()[0]->inputs()[0]->name()=="transpose"&& // transpose
                ((ins->inputs()[0]->inputs()[0]->inputs()[0]->name()=="gpu::add"&& // conv(可能由偏置)
                ins->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->name()=="gpu::convolution")||
                ins->inputs()[0]->inputs()[0]->inputs()[0]->name()=="gpu::convolution")&&
                ins->outputs()[0]->name()=="gpu::concat" // concat
                )
        {
            
            if(dims[0]!=ins->inputs()[0]->get_shape().lens()[0])
            {
                dims[0]=ins->inputs()[0]->get_shape().lens()[0];
            }

        }
        /*** YOLOV5模型中的动态shape计算图模式
         * 
         * 
         计算图示例：
         main:@140 = gpu::convolution[padding={0, 0, 0, 0},stride={1, 1},dilation={1, 1},group=1,padding_mode=0,solution_id=128](main:@137,main:@131,main:@139,main:@138) -> float_type, {1, 255, 26, 26}, {172380, 676, 26, 1}
         main:@144 = gpu::add(main:@140,main:@143,main:@142) -> float_type, {1, 255, 26, 26}, {172380, 676, 26, 1}
         main:@150 = reshape[dims={1, 3, 85, 26, 26}](main:@144) -> float_type, {1, 3, 85, 26, 26}, {172380, 57460, 676, 26, 1}
         main:@151 = transpose[permutation={0, 1, 3, 4, 2}](main:@150) -> float_type, {1, 3, 26, 26, 85}, {172380, 57460, 26, 1, 676}
         main:@152 = gpu::sigmoid(main:@151,main:@149) -> float_type, {1, 3, 26, 26, 85}, {172380, 57460, 26, 1, 676}
         main:@187 = gpu::contiguous(main:@151,main:@186) -> float_type, {1, 3, 26, 26, 85}, {172380, 57460, 2210, 85, 1}
         main:@188 = hip::copy_from_gpu[shape=float_type, {1, 3, 26, 26, 85}, {172380, 57460, 2210, 85, 1},id=hip::copy_from_gpu1](main:@187) -> float_type, {1, 3, 26, 26, 85}, {172380, 57460, 2210, 85, 1}
         main:@217 = hip::sync_stream(main:@216,main:@188,main:@166) -> float_type, {1, 2535, 85}, {215475, 85, 1}
         * 
        */
        else if(
                ((ins->inputs()[0]->name()=="gpu::add"&& 
                ins->inputs()[0]->inputs()[0]->name()=="gpu::convolution")||
                ins->inputs()[0]->name()=="gpu::convolution")&&
                ins->outputs()[0]->name()=="transpose" &&
                (ins->outputs()[0]->outputs()[0]->name()=="gpu::sigmoid"||ins->outputs()[0]->outputs()[0]->name()=="gpu::contiguous")&&
                dims.size()==5
                )
        {
            // dims[1]，dims[2]两个维度为固定值
            std::vector<std::size_t> lens=ins->inputs()[0]->get_shape().lens();
            dims[0]=lens[0];
            dims[3]=lens[2];
            dims[4]=lens[3];
        }
        /*** YOLOV5模型中的动态shape计算图模式2
         * 
         * 
         计算图示例：
         main:@697 = gpu::mul(main:@693,main:@695,main:@696) -> float_type, {8, 3, 10, 10, 2}, {600, 200, 20, 2, 1}
         main:@702 = gpu::mul_add_mul(main:@699,main:@689,main:@701,main:@700,main:@698) -> float_type, {8, 3, 10, 10, 2}, {600, 200, 20, 2, 1}
         main:@706 = slice[axes={4},starts={4},ends={85}](main:@644) -> float_type, {8, 3, 10, 10, 81}, {25500, 8500, 10, 1, 100}
         main:@708 = gpu::concat[axis=4](main:@702,main:@697,main:@706,main:@707) -> float_type, {8, 3, 10, 10, 85}, {25500, 8500, 850, 85, 1}
         main:@709 = reshape[dims={8, 300, 85}](main:@708) -> float_type, {8, 300, 85}, {25500, 85, 1}
         main:@713 = gpu::concat[axis=1](main:@712,main:@710,main:@709,main:@711) -> float_type, {8, 6300, 85}, {535500, 85, 1}
         * 
        */
        else if(
                dims.size()==3&&
                ins->inputs()[0]->name()=="gpu::concat"&&
                ins->inputs()[0]->get_shape().lens().size()==5&&
                ins->inputs()[0]->inputs().size()==4&&
                ins->inputs()[0]->inputs()[1]->name()=="gpu::mul"&&
                ins->inputs()[0]->inputs()[2]->name()=="slice"&&
                ins->outputs()[0]->name()=="gpu::concat"&&
                ins->outputs()[0]->get_shape().lens().size()==3
                // ins->outputs()[0]->inputs().size()==4&& // 不能加入这些限制，有可能会有多个检测层
                // ins->outputs()[0]->inputs()[0]->name()=="reshape"&&
                // ins->outputs()[0]->inputs()[1]->name()=="reshape"&&
                // ins->outputs()[0]->inputs()[2]->name()=="reshape"
                )
        {
            std::vector<std::size_t> lens=ins->inputs()[0]->get_shape().lens();
            dims[0]=lens[0];
            dims[1]=lens[1]*lens[2]*lens[3];
            dims[2]=lens[4];
        }
        
       /*** GPT2_MIGraphXSamples模型中的动态shape计算图模式1:attention中计算qkv后的reshape
         * 
         * 
        计算图示例：当输入shape为 int64_type, {1, 256}, {256, 1}
        main:@50 = hip::hip_copy_literal[id=main:@literal:142] -> float_type, {768, 2304}, {2304, 1}
        main:@49 = gpu::div_mul_add(main:@29,main:@48,main:@47,main:@46,main:@45) -> float_type, {1, 256, 768}, {196608, 768, 1}
        main:@52 = reshape[dims={-1, 768}](main:@49) -> float_type, {256, 768}, {768, 1}
        main:@53 = gpu::gemm[alpha=1,beta=1,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@52,main:@50,main:@30,main:@51) -> float_type, {256, 2304}, {2304, 1}
        main:@54 = reshape[dims={1, 256, 2304}](main:@53) -> float_type, {1, 256, 2304}, {589824, 2304, 1} //
        main:@55 = reshape[dims={1, 256, 36, 64}](main:@54) -> float_type, {1, 256, 36, 64}, {589824, 2304, 64, 1}
        main:@56 = transpose[permutation={0, 2, 1, 3}](main:@55) -> float_type, {1, 36, 256, 64}, {589824, 64, 2304, 1}
        main:@57 = slice[axes={1},starts={0},ends={12}](main:@56) -> float_type, {1, 12, 256, 64}, {589824, 64, 2304, 1}
        main:@58 = load[offset=5505024,end=8650752](main:@1) -> float_type, {1, 12, 256, 256}, {786432, 65536, 256, 1}
        main:@59 = slice[axes={2},starts={12},ends={24}](main:@55) -> float_type, {1, 256, 12, 64}, {589824, 2304, 64, 1}
        main:@60 = transpose[permutation={0, 2, 3, 1}](main:@59) -> float_type, {1, 12, 64, 256}, {589824, 64, 1, 2304}
        main:@61 = gpu::gemm[alpha=1,beta=0,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@57,main:@60,main:@58) -> float_type, {1, 12, 256, 256}, {786432, 65536, 256, 1}
        main:@62 = hip::hip_copy_literal[id=main:@literal:62] -> float_type, {1}, {0}
        main:@63 = hip::hip_copy_literal[id=main:@literal:42] -> bool_type, {1, 1, 256, 256}, {65536, 65536, 256, 1}
        main:@64 = load[offset=2359296,end=4718592](main:@1) -> float_type, {1, 36, 256, 64}, {589824, 16384, 64, 1}
        main:@65 = gpu::contiguous(main:@56,main:@64) -> float_type, {1, 36, 256, 64}, {589824, 16384, 64, 1}
        main:@66 = hip::hip_copy_literal[id=main:@literal:11] -> float_type, {1}, {0}
        main:@67 = load[offset=8650752,end=11796480](main:@1) -> float_type, {1, 12, 256, 256}, {786432, 65536, 256, 1}
        main:@68 = multibroadcast[out_lens={1, 12, 256, 256}](main:@66) -> float_type, {1, 12, 256, 256}, {0, 0, 0, 0}
        main:@69 = gpu::mul(main:@61,main:@68,main:@67) -> float_type, {1, 12, 256, 256}, {786432, 65536, 256, 1}
        main:@70 = load[offset=11796480,end=14942208](main:@1) -> float_type, {1, 12, 256, 256}, {786432, 65536, 256, 1}
        main:@71 = multibroadcast[out_lens={1, 12, 256, 256}](main:@63) -> bool_type, {1, 12, 256, 256}, {65536, 0, 256, 1}
        main:@72 = multibroadcast[out_lens={1, 12, 256, 256}](main:@62) -> float_type, {1, 12, 256, 256}, {0, 0, 0, 0}
        main:@73 = gpu::where(main:@71,main:@69,main:@72,main:@70) -> float_type, {1, 12, 256, 256}, {786432, 65536, 256, 1}
        main:@74 = load[offset=6291456,end=9437184](main:@1) -> float_type, {1, 12, 256, 256}, {786432, 65536, 256, 1}
        main:@75 = gpu::softmax[axis=3](main:@73,main:@74) -> float_type, {1, 12, 256, 256}, {786432, 65536, 256, 1}
        main:@76 = load[offset=5505024,end=6291456](main:@1) -> float_type, {1, 256, 12, 64}, {196608, 768, 64, 1}
        main:@77 = transpose[permutation={0, 2, 1, 3}](main:@76) -> float_type, {1, 12, 256, 64}, {196608, 64, 768, 1}
        main:@78 = slice[axes={1},starts={24},ends={36}](main:@65) -> float_type, {1, 12, 256, 64}, {589824, 16384, 64, 1}
        main:@79 = gpu::gemm[alpha=1,beta=0,int8_x4_format=1,compute_fp32=0,trans_batch=1](main:@75,main:@78,main:@77) -> float_type, {1, 12, 256, 64}, {196608, 64, 768, 1}
        main:@80 = hip::hip_copy_literal[id=main:@literal:31] -> float_type, {256, 768}, {768, 1}
        main:@81 = hip::hip_copy_literal[id=main:@literal:141] -> float_type, {768, 768}, {768, 1}
        main:@82 = transpose[permutation={0, 2, 1, 3}](main:@79) -> float_type, {1, 256, 12, 64}, {196608, 768, 64, 1}
        main:@83 = load[offset=6291456,end=7077888](main:@1) -> float_type, {256, 768}, {768, 1}
        main:@84 = reshape[dims={1, 256, 768}](main:@82) -> float_type, {1, 256, 768}, {196608, 768, 1}
        main:@85 = reshape[dims={-1, 768}](main:@84) -> float_type, {256, 768}, {768, 1}
        main:@86 = gpu::gemm[alpha=1,beta=1,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@85,main:@81,main:@80,main:@83) -> float_type, {256, 768}, {768, 1}
        main:@87 = load[offset=5505024,end=6291456](main:@1) -> float_type, {1, 256, 768}, {196608, 768, 1}
        main:@88 = reshape[dims={1, 256, 768}](main:@86) -> float_type, {1, 256, 768}, {196608, 768, 1}
        main:@89 = gpu::add(main:@88,main:@16,main:@87) -> float_type, {1, 256, 768}, {196608, 768, 1}
        main:@90 = load[offset=4718592,end=5505024](main:@1) -> float_type, {1, 256, 768}, {196608, 768, 1}
        main:@91 = gpu::mul(main:@89,main:@20,main:@90) -> float_type, {1, 256, 768}, {196608, 768, 1}
        main:@92 = load[offset=6292480,end=6293504](main:@1) -> float_type, {1, 256, 1}, {256, 1, 1}
        main:@93 = gpu::reduce_mean[axes={2}](main:@91,main:@92) -> float_type, {1, 256, 1}, {256, 1, 1}
         * 
        */
       else if(
            num_neg==0&&
            dims.size()>ins->inputs()[0]->get_shape().lens().size()&&
            dims[dims.size()-1]==inputs[0]->get_shape().lens()[inputs[0]->get_shape().lens().size()-1]&&
            ins->inputs()[0]->name()=="gpu::gemm"&&
            ins->outputs()[0]->name()=="reshape"&&
            ins->outputs()[0]->get_shape().lens().size()>ins->get_shape().lens().size()
              )
        {
            std::vector<std::size_t> lens=ins->inputs()[0]->inputs()[0]->inputs()[0]->get_shape().lens();
            dims[0]=lens[0];
            dims[1]=lens[1];
            any_cast<migraphx::op::reshape>(ins->outputs()[0]->get_operator()).dims[0]=lens[0];
            any_cast<migraphx::op::reshape>(ins->outputs()[0]->get_operator()).dims[1]=lens[1];

        }
        /*** GPT2_MIGraphXSamples模型中的动态shape计算图模式2: attention中最后一个gemm计算完成后的reshape
         * 
         * 
        */
       else if(
            num_neg==0&&
            ins->inputs()[0]->name()=="gpu::gemm"&&
            dims[dims.size()-1]==inputs[0]->get_shape().lens()[inputs[0]->get_shape().lens().size()-1]&&
            ins->outputs()[0]->name()=="gpu::add"&&
            ins->outputs()[0]->outputs().size()==3
             )
        {
            std::vector<std::size_t> lens=ins->inputs()[0]->inputs()[0]->inputs()[0]->get_shape().lens();
            dims[0]=lens[0];
            dims[1]=lens[1];

        }
        /*** GPT2_MIGraphXSamples模型中的动态shape计算图模式3
         * 
         * div_mul_add/mul->gemm->reshape
         * 
         * 
        */
        else if(
            num_neg==0&&
            ins->inputs()[0]->name()=="gpu::gemm"&&
            ins->inputs()[0]->inputs()[0]->name()=="reshape"&&
            (ins->inputs()[0]->inputs()[0]->inputs()[0]->name()=="gpu::div_mul_add"||ins->inputs()[0]->inputs()[0]->inputs()[0]->name()=="gpu::mul")&&
            dims[dims.size()-1]==inputs[0]->get_shape().lens()[inputs[0]->get_shape().lens().size()-1]
             )
        {
            std::vector<std::size_t> lens=ins->inputs()[0]->inputs()[0]->inputs()[0]->get_shape().lens();
            dims[0]=lens[0];
            dims[1]=lens[1];

        }
        /*** 常规attention中动态shape计算图模式：第一个gemm(计算qkv)后面的reshape
         * 
         * 
        计算图示例：
        main:@57 = gpu::gemm[alpha=1,beta=0,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@55,main:@34,main:@56) -> float_type, {1, 197, 2304}, {453888, 2304, 1} // 计算qkv
        main:@58 = hip::hip_copy_literal[id=main:@literal:153] -> float_type, {2304}, {1}
        main:@59 = load[offset=605184,end=2420736](main:@1) -> float_type, {1, 197, 2304}, {453888, 2304, 1}
        main:@60 = multibroadcast[out_lens={1, 197, 2304}](main:@58) -> float_type, {1, 197, 2304}, {0, 0, 1}
        main:@61 = gpu::add(main:@60,main:@57,main:@59) -> float_type, {1, 197, 2304}, {453888, 2304, 1} // 加上偏置
        main:@62 = hip::hip_copy_literal[id=main:@literal:50] -> float_type, {1}, {0}
        main:@63 = hip::hip_copy_literal[id=main:@literal:51] -> float_type, {1}, {0}
        main:@64 = load[offset=2420736,end=3025920](main:@1) -> float_type, {1, 12, 197, 64}, {151296, 12608, 64, 1}
        main:@65 = reshape[dims={1, 197, 3, 12, 64}](main:@61) -> float_type, {1, 197, 3, 12, 64}, {453888, 2304, 768, 64, 1} // 执行reshape->{1, 197, 3, 12, 64},12个头，每个头64维
        main:@66 = transpose[permutation={2, 0, 3, 1, 4}](main:@65) -> float_type, {3, 1, 12, 197, 64}, {768, 453888, 64, 2304, 1} // 执行transpose->{3, 1, 12, 197, 64}
        */
        else if(
                num_neg==0&&
                dims.size()>ins->inputs()[0]->get_shape().lens().size()&&
                (ins->inputs()[0]->name()=="gpu::add"||ins->inputs()[0]->name()=="gpu::gemm")&&
                ins->outputs()[0]->name()=="transpose"&&
                ins->outputs()[0]->outputs().size()>=2 // 需要使用三个gather或者slice操作从第一个gemm的结果中得到q,k,v
                )
        {
            if(ins->inputs()[0]->get_shape().lens().size()==3)
            {
                dims[0]=ins->inputs()[0]->get_shape().lens()[0];
                dims[1]=ins->inputs()[0]->get_shape().lens()[1];

            }
            else if(ins->inputs()[0]->get_shape().lens().size()==2)
            {
                dims[1]=ins->inputs()[0]->get_shape().lens()[0];
            }
            
        }
        /*** 常规attention中动态shape计算图模式2：attention中倒数第二个gemm后面的reshape
         * 
        */
        else if(
                num_neg==0&&
                dims.size()<ins->inputs()[0]->get_shape().lens().size()&&
                ((ins->inputs()[0]->name()=="transpose")||(ins->inputs()[0]->name()=="gpu::contiguous"&&ins->inputs()[0]->inputs()[0]->name()=="transpose"))&&
                ((ins->outputs()[0]->name()=="gpu::gemm")||(ins->outputs()[0]->name()=="reshape"&&ins->outputs()[0]->outputs()[0]->name()=="gpu::gemm"))
                )
        {
            if(dims.size()==3)
            {
                dims[0]=ins->inputs()[0]->get_shape().lens()[0];
                dims[1]=ins->inputs()[0]->get_shape().lens()[1];
            }
            else if(dims.size()==2)
            {
                dims[0]=ins->inputs()[0]->get_shape().lens()[0]*ins->inputs()[0]->get_shape().lens()[1];
            }

        }
        /*** LSTM模型的动态shape模式

        计算图示例：
        main:@760 = hip::hip_copy_literal[id=main:@literal:57] -> float_type, {1, 192, 48}, {9216, 48, 1} // hidden_size:48
        main:@775 = squeeze[axes={1}](main:@774) -> float_type, {80, 8, 48}, {384, 48, 1} // sequenceLen:80,batchsize:8
        main:@777 = gpu::lstm[hidden_size=48,actv_func={sigmoid, tanh, tanh},direction=forward,clip=0,input_forget=0](main:@775,main:@756,main:@760,main:@754,main:@762,main:@740,main:@740,main:@762,main:@776) -> float_type, {80, 1, 8, 48}, {384, 384, 48, 1}
        main:@782 = squeeze[axes={1}](main:@777) -> float_type, {80, 8, 48}, {384, 48, 1}
        main:@784 = gpu::concat[axis=2](main:@782,main:@781,main:@783) -> float_type, {80, 8, 144}, {1152, 144, 1}
        main:@785 = reshape[dims={640, 144}](main:@784) -> float_type, {640, 144}, {144, 1}
        main:@787 = gpu::gemm[alpha=1,beta=1,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@785,main:@755,main:@749,main:@786) -> float_type, {640, 5531}, {5531, 1}
        main:@788 = reshape[dims={80, 8, -1}](main:@787) -> float_type, {80, 8, 5531}, {44248, 5531, 1}
        */
        else if (hasLSTM&&num_neg==1)
        {
            int sequenceLen = lstm_ins->inputs()[0]->get_shape().lens()[0];
            int batchsize = lstm_ins->inputs()[0]->get_shape().lens()[1];
            int hidden_size = lstm_ins->inputs()[2]->get_shape().lens()[2];

            // 修改reshape算子的属性
            dims[0]=sequenceLen;
            dims[1]=batchsize;
        }
        /*** SVTR模型动态shape计算图模式
         * 
         * 
        计算图示例：
        main:@556 = gpu::sigmoid_mul(main:@554,main:@555) -> float_type, {2, 120, 1, 32}, {3840, 32, 32, 1}
        main:@561 = reshape[dims={2, 120, -1}](main:@556) -> float_type, {2, 120, 32}, {3840, 32, 1}
        main:@562 = transpose[permutation={0, 2, 1}](main:@561) -> float_type, {2, 32, 120}, {3840, 1, 32}
        main:@563 = load[offset=161792,end=192512](main:@1) -> float_type, {2, 32, 120}, {3840, 120, 1}
        main:@564 = gpu::mul(main:@562,main:@560,main:@563) -> float_type, {2, 32, 120}, {3840, 120, 1}
        main:@565 = load[offset=192768,end=193024](main:@1) -> float_type, {2, 32, 1}, {32, 1, 1}
        main:@566 = gpu::reduce_mean[axes={2}](main:@564,main:@565) -> float_type, {2, 32, 1}, {32, 1, 1}
         * 
        */
        else if(
                num_neg==1&&
                ins->inputs()[0]->name()=="gpu::sigmoid_mul"&&
                ins->outputs()[0]->name()=="transpose"&&
                ins->outputs()[0]->outputs().size()==3&&
                ins->outputs()[0]->outputs()[0]->name()=="gpu::mul"&&
                ins->outputs()[0]->outputs()[0]->outputs()[0]->name()=="gpu::reduce_mean"&&
                dims.size()==3
               )
        {
            dims[0]=ins->inputs()[0]->get_shape().lens()[0];
            dims[1]=ins->inputs()[0]->get_shape().lens()[1];

        }
        /*** crnn_lite_lstm_Nx3x32xN模型动态shape计算图模式
         * 
         * 
        计算图示例：
        main:@772 = gpu::lstm[hidden_size=48,actv_func={sigmoid, tanh, tanh, sigmoid, tanh, tanh},direction=bidirectional,clip=0,input_forget=0](main:@770,main:@753,main:@749,main:@748,main:@766,main:@747,main:@747,main:@766,main:@771) -> float_type, {256, 2, 1, 48}, {96, 48, 48, 1}
        main:@779 = transpose[permutation={0, 2, 1, 3}](main:@772) -> float_type, {256, 1, 2, 48}, {96, 48, 48, 1}
        main:@780 = reshape[dims={0, 0, -1}](main:@779) -> float_type, {256, 1, 96}, {96, 96, 1}
        main:@777 = gpu::lstm[hidden_size=48,actv_func={sigmoid, tanh, tanh},direction=forward,clip=0,input_forget=0](main:@775,main:@751,main:@757,main:@746,main:@766,main:@756,main:@756,main:@766,main:@776) -> float_type, {256, 1, 1, 48}, {48, 48, 48, 1}
        main:@781 = squeeze[axes={1}](main:@777) -> float_type, {256, 1, 48}, {48, 48, 1}
        main:@782 = gpu::concat[axis=2](main:@781,main:@780,main:@778) -> float_type, {256, 1, 144}, {144, 144, 1}
        main:@783 = reshape[dims={256, 144}](main:@782) -> float_type, {256, 144}, {144, 1}
        main:@784 = load[offset=245760,end=5909504](main:@1) -> float_type, {256, 5531}, {5531, 1}
        main:@785 = gpu::gemm[alpha=1,beta=1,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@783,main:@760,main:@739,main:@784) -> float_type, {256, 5531}, {5531, 1}
        main:@786 = reshape[dims={256, 1, -1}](main:@785) -> float_type, {256, 1, 5531}, {5531, 5531, 1}

        batchsize=2的时候，main:@780指令后面会有gpu::contiguous
        main:@779 = gpu::lstm[hidden_size=48,actv_func={sigmoid, tanh, tanh},direction=forward,clip=0,input_forget=0](main:@777,main:@755,main:@754,main:@753,main:@766,main:@752,main:@752,main:@766,main:@778) -> float_type, {256, 1, 2, 48}, {96, 96, 48, 1}
        main:@780 = transpose[permutation={0, 2, 1, 3}](main:@774) -> float_type, {256, 2, 2, 48}, {192, 48, 96, 1}
        main:@781 = load[offset=393216,end=589824](main:@1) -> float_type, {256, 2, 2, 48}, {192, 96, 48, 1}
        main:@782 = gpu::contiguous(main:@780,main:@781) -> float_type, {256, 2, 2, 48}, {192, 96, 48, 1}
        main:@783 = load[offset=98304,end=393216](main:@1) -> float_type, {256, 2, 144}, {288, 144, 1}
        main:@784 = reshape[dims={0, 0, -1}](main:@782) -> float_type, {256, 2, 96}, {192, 96, 1}
        main:@785 = squeeze[axes={1}](main:@779) -> float_type, {256, 2, 48}, {96, 48, 1}
        main:@786 = gpu::concat[axis=2](main:@785,main:@784,main:@783) -> float_type, {256, 2, 144}, {288, 144, 1}
        main:@787 = load[offset=393216,end=11720704](main:@1) -> float_type, {512, 5531}, {5531, 1}
        main:@788 = reshape[dims={512, 144}](main:@786) -> float_type, {512, 144}, {144, 1}
        main:@789 = gpu::gemm[alpha=1,beta=1,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@788,main:@762,main:@764,main:@787) -> float_type, {512, 5531}, {5531, 1}
        main:@790 = reshape[dims={256, 2, -1}](main:@789) -> float_type, {256, 2, 5531}, {11062, 5531, 1}
         * 
        */
        else if(
                ins->inputs()[0]->name()=="gpu::concat"&&
                ins->inputs()[0]->inputs()[0]->name()=="squeeze"&&
                ins->inputs()[0]->inputs()[0]->inputs()[0]->name()=="gpu::lstm"&&
                ins->inputs()[0]->inputs()[1]->name()=="reshape"&&
                ins->outputs()[0]->name()=="gpu::gemm"&&
                ins->outputs()[0]->outputs()[0]->name()=="reshape"
               )
        {
            // 修改第一个reshape
            dims[0]=ins->inputs()[0]->get_shape().lens()[0]*ins->inputs()[0]->get_shape().lens()[1];
            dims[1]=ins->inputs()[0]->get_shape().lens()[2];

            // 修改第二个reshape
            instruction_ref second_reshape=ins->outputs()[0]->outputs()[0];
            any_cast<migraphx::op::reshape>(second_reshape->get_operator()).dims[0]=ins->inputs()[0]->get_shape().lens()[0];
            any_cast<migraphx::op::reshape>(second_reshape->get_operator()).dims[1]=ins->inputs()[0]->get_shape().lens()[1];

        }
        /*** crnn_lite_lstm_Nx3x32xN模型动态shape计算图模式2
         * 
         * 
        计算图示例：
        main:@34 = gpu::convolution[padding={1, 1, 1, 1},stride={2, 1},dilation={1, 1},group=8,padding_mode=0,solution_id=125](main:@30,main:@31,main:@33,main:@32) -> float_type, {2, 8, 8, 512}, {32768, 4096, 512, 1}
        main:@40 = gpu::add(main:@34,main:@39,main:@38) -> float_type, {2, 8, 8, 512}, {32768, 4096, 512, 1}
        main:@42 = reshape[dims={16, -1}](main:@40) -> float_type, {16, 4096}, {4096, 1}
        main:@44 = gpu::reduce_mean[axes={1}](main:@42,main:@43) -> float_type, {16, 1}, {1, 1}
        main:@46 = reshape[dims={2, 8, 1, 1}](main:@44) -> float_type, {2, 8, 1, 1}, {8, 1, 1, 1}
        main:@47 = reshape[dims={2, 8}](main:@46) -> float_type, {2, 8}, {8, 1}
        main:@48 = gpu::gemm[alpha=1,beta=0,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@47,main:@41,main:@45) -> float_type, {2, 2}, {2, 1}
        main:@51 = gpu::add_relu(main:@48,main:@49,main:@50) -> float_type, {2, 2}, {2, 1}
        main:@55 = gpu::gemm[alpha=1,beta=1,int8_x4_format=1,compute_fp32=0,trans_batch=0](main:@51,main:@52,main:@53,main:@54) -> float_type, {2, 8}, {8, 1}
        main:@59 = reshape[dims={2, 8, 1, 1}](main:@55) -> float_type, {2, 8, 1, 1}, {8, 1, 1, 1} // 匹配该reshape
         * 
        */
        else if(
                ins->inputs()[0]->name()=="gpu::gemm"&&
                ins->inputs()[0]->inputs()[0]->name()=="gpu::add_relu"&&
                ins->inputs()[0]->inputs()[0]->inputs()[0]->name()=="gpu::gemm"&&
                ins->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->name()=="reshape"&&
                ins->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->name()=="reshape"&&
                ins->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->name()=="gpu::reduce_mean"&&
                ins->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->name()=="reshape"
               )
        {
            std::vector<std::size_t> lens=ins->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->get_shape().lens();
            dims[0]=lens[0];
            dims[1]=lens[1];
        }
        else
        {
        }

        return true;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
