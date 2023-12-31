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
#include <migraphx/add_contiguous_for_dynamic.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/*
在动态shape中，对于需要standard或者packed输入的指令，如果其输入参数是slice或者multibroadcast这些能够产生view的指令，则需要添加contiguous

比如对于ModelZoo/Dynamic/GPT2_MIGraphXSamples_1xN.onnx中的attention模块存在如下计算图：

main:@208 = hip::hip_copy_literal[id=main:@literal:175] -> uint8_type, {1, 1, 300, 300}, {90000, 90000, 300, 1} // mask矩阵
main:@217 = gpu::slice_dynamic[axes={2},max_starts={0},max_ends={300}](main:@208,main:@216,main:@215,main:@210,main:@209) -> uint8_type, {1, 1, 300, 300}, {90000, 90000, 300, 1}
main:@218 = gpu::slice_dynamic[axes={3},max_starts={0},max_ends={300}](main:@217,main:@25,main:@215,main:@189,main:@209) -> uint8_type, {1, 1, 300, 300}, {90000, 90000, 300, 1}
main:@219 = load[offset=2764824,end=2854824](main:@1) -> bool_type, {1, 1, 300, 300}, {90000, 90000, 300, 1}
main:@220 = gpu::convert[target_type=0](main:@218,main:@219) -> bool_type, {1, 1, 300, 300}, {90000, 90000, 300, 1}
main:@221 = gpu::multibroadcast_dynamic[max_out_lens={1, 12, 300, 300}](main:@188,main:@183) -> float_type, {1, 12, 300, 300}, {0, 0, 0, 0}
main:@222 = gpu::multibroadcast_dynamic[max_out_lens={1, 12, 300, 300}](main:@220,main:@183) -> bool_type, {1, 12, 300, 300}, {90000, 0, 300, 1}
main:@223 = load[offset=9849624,end=14169624](main:@1) -> float_type, {1, 12, 300, 300}, {1080000, 90000, 300, 1}
main:@224 = gpu::where(main:@222,main:@176,main:@221,main:@223) -> float_type, {1, 12, 300, 300}, {1080000, 90000, 300, 1}
main:@220的convert需要输入是packed的，但是其输入是一个slice，在动态shape中，其输入有可能是packed的，也有可能不是packed，则需要统一加入contiguous

*/
void add_contiguous_for_dynamic::apply(module& m) const
{
    std::string key1 = "require_std_shape";
    std::string key2 = "require_packed_shape";
    for(auto ins : iterator_for(m))
    {
        auto&& attr = ins->get_operator().attributes();
        if((attr.get(key1, false))||(attr.get(key2, false)))
        {
            auto args     = ins->inputs();
            auto new_args = args;
            std::transform(args.begin(), args.end(), new_args.begin(), [&](auto in) 
            {
                /*
                对于一个需要standard输入的指令，如果输入是一个slice或者multibroadcast，则说明slice或者multibroadcast在最大shape中产生的是一个standard的输出
                但是在动态shape中，其他shape就有可能产生非standard，所以需要加入contiguous
                */
                if(in->name() == "gpu::slice_dynamic"||
                    in->name() == "slice"||
                    in->name() == "gpu::multibroadcast_dynamic"||
                    in->name() == "multibroadcast")
                {
                    auto allocate_ins=m.insert_instruction(in,make_op("allocate", {{"shape", to_value(shape{in->get_shape().type(),in->get_shape().lens()})}}));
                    
                    return m.insert_instruction(ins, make_op("gpu::contiguous"), in,allocate_ins);
                }
                else
                {
                    return in;
                }
                
            });

            if(new_args != args)
            {
                m.replace_instruction(ins, ins->get_operator(), new_args);
            }
        }
    }

}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
