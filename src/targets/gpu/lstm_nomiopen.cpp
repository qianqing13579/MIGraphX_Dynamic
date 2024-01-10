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
#include <migraphx/gpu/lstm_nomiopen.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/lstm.hpp>
#include <migraphx/gpu/device/contiguous.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape lstm_nomiopen::compute_shape(const std::vector<shape>& inputs) const
{
    std::vector<shape> inputs2;
    inputs2.push_back(inputs[0]);
    check_shapes{inputs2, *this}.standard();
    return op.compute_shape(inputs);
}

/*
args[0]:输入tensor
args[1]:W,input weight matrix(输入权值)
args[2]:R,hidden state weight matrix(即隐藏层的权值)
args[3]:B,bias
args[4]:初始隐藏层状态，intial hidden state
args[5]:初始cell状态，initial cell value
*/
argument lstm_nomiopen::compute(context& ctx,
                                const shape& output_shape,
                                const std::vector<argument>& args) const
{
    int num_directions = static_cast<int>(op.direction) >= 2 ? 2 : 1;

    device::LSTM(ctx,
                 args[args.size() - 1],
                 args[0],
                 W,
                 R,
                 args[3],
                 h_0,
                 c_0,
                 h_t,
                 c_t,
                 XW,
                 x_reverse,
                 HR,
                 result_gpu,
                 num_directions);

    return args[args.size() - 1];
}

void lstm_nomiopen::finalize(context& ctx, const shape& output_shape, std::vector<shape> inputs)
{
    if(is_first_finalize)
    {
        // 获取属性
        int sequenceLen    = inputs[0].lens()[0]; // 输入数据的序列长度
        int batchsize      = inputs[0].lens()[1]; // batchsize
        int input_size     = inputs[0].lens()[2]; // 输入数据的特征维度
        int hidden_size    = inputs[2].lens()[2]; // 隐藏层特征维度
        int num_directions = static_cast<int>(op.direction) >= 2 ? 2 : 1;

        // initial_h,如果没有设置则初始化为0: [num_directions, batch_size, hidden_size]
        h_0 = allocate_gpu(
            shape{inputs[0].type(), {(std::size_t)num_directions * batchsize * hidden_size}});
        if(inputs.size() >= 5)
        {
            if(args2.size() >= 4)
            {
                h_0 = to_gpu(args2[3]);
            }
            else
            {
                hipMemset(h_0.data(), 0, h_0.get_shape().bytes());
            }
        }
        else
        {
            hipMemset(h_0.data(), 0, h_0.get_shape().bytes());
        }

        // initial_c,如果没有设置则初始化为0: shape [num_directions, batch_size, hidden_size]
        c_0 = allocate_gpu(
            shape{inputs[0].type(), {(std::size_t)num_directions * batchsize * hidden_size}});
        if(inputs.size() >= 6)
        {
            if(args2.size() >= 5)
            {
                c_0 = to_gpu(args2[4]);
            }
            else
            {
                hipMemset(c_0.data(), 0, c_0.get_shape().bytes());
            }
        }
        else
        {
            hipMemset(c_0.data(), 0, c_0.get_shape().bytes());
        }

        // 分配临时buffer
        result_gpu = allocate_gpu(
            shape{inputs[0].type(),
                  {(std::size_t)num_directions * sequenceLen * batchsize * hidden_size}});
        XW = allocate_gpu(
            shape{inputs[0].type(),
                  {(std::size_t)sequenceLen * batchsize, (std::size_t)4 * hidden_size}});
        x_reverse = allocate_gpu(shape{
            inputs[0].type(), {(std::size_t)sequenceLen * batchsize, (std::size_t)input_size}});
        HR        = allocate_gpu(
            shape{inputs[0].type(), {(std::size_t)batchsize, (std::size_t)4 * hidden_size}});
        h_t = allocate_gpu(
            shape{inputs[0].type(), {(std::size_t)num_directions * batchsize * hidden_size}});
        c_t = allocate_gpu(
            shape{inputs[0].type(), {(std::size_t)num_directions * batchsize * hidden_size}});

        // 对W进行转置
        shape s_w    = args2[0].get_shape();
        argument w   = to_gpu(args2[0]);
        argument w_1 = w.reshape(shape{s_w.type(),
                                       {s_w.lens()[0], s_w.lens()[2], s_w.lens()[1]},
                                       {s_w.strides()[0], s_w.strides()[2], s_w.strides()[1]}});
        W = allocate_gpu(shape{s_w.type(), {s_w.lens()[0], s_w.lens()[2], s_w.lens()[1]}});
        device::contiguous(ctx.get_stream().get(), W, w_1);

        // 对R进行转置
        shape s_r    = args2[1].get_shape();
        argument r   = to_gpu(args2[1]);
        argument r_1 = r.reshape(shape{s_r.type(),
                                       {s_r.lens()[0], s_r.lens()[2], s_r.lens()[1]},
                                       {s_r.strides()[0], s_r.strides()[2], s_r.strides()[1]}});
        R = allocate_gpu(shape{s_r.type(), {s_r.lens()[0], s_r.lens()[2], s_r.lens()[1]}});
        device::contiguous(ctx.get_stream().get(), R, r_1);

        is_first_finalize = false;
    }
}
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
