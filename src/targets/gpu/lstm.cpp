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
#include <migraphx/gpu/lstm.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/SimpleLog.h>
#include <migraphx/gpu/device/convert.hpp>
#include <migraphx/gpu/device/contiguous.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// 将onnx权值转换为MIOpen格式
template <typename DataType>
static void convert_onnx_to_MIOpen(context& ctx,
                                   const migraphx::argument& W,
                                   const migraphx::argument& R,
                                   const migraphx::argument& B,
                                   bool has_bias,
                                   const int directions,
                                   const int hidden_size,
                                   const int input_size,
                                   DataType* weight_ptr)
{
    // IOFC->IFOC
    const int gate_offset[4] = {0, 2, 1, 3};

    // [num_directions, 4*hidden_size, input_size].
    DataType* W_ptr = (DataType*)W.data();
    // [num_directions, 4*hidden_size, hidden_size].
    DataType* R_ptr = (DataType*)R.data();
    // [num_directions, 8*hidden_size].
    DataType* B_ptr = has_bias ? (DataType*)B.data() : nullptr;

    size_t offset = 0;

    // W
    for(int dire = 0; dire < directions; dire++)
    {
        for(int g = 0; g < 4; g++)
        {
            hipMemcpyAsync(weight_ptr + offset,
                           W_ptr + (dire * 4 + gate_offset[g]) * hidden_size * input_size,
                           hidden_size * input_size * sizeof(DataType),
                           hipMemcpyDeviceToDevice,
                           ctx.get_stream().get());
            offset += hidden_size * input_size;
        }
    }

    // R
    for(int dire = 0; dire < directions; dire++)
    {
        for(int g = 0; g < 4; g++)
        {
            hipMemcpyAsync(weight_ptr + offset,
                           R_ptr + (dire * 4 + gate_offset[g]) * hidden_size * hidden_size,
                           hidden_size * hidden_size * sizeof(DataType),
                           hipMemcpyDeviceToDevice,
                           ctx.get_stream().get());
            offset += hidden_size * hidden_size;
        }
    }

    if(has_bias)
    {
        // WB
        for(int dire = 0; dire < directions; dire++)
        {
            for(int g = 0; g < 4; g++)
            {
                hipMemcpyAsync(weight_ptr + offset,
                               B_ptr + (dire * 8 + gate_offset[g]) * hidden_size,
                               hidden_size * sizeof(DataType),
                               hipMemcpyDeviceToDevice,
                               ctx.get_stream().get());
                offset += hidden_size;
            }
        }

        // RB
        for(int dire = 0; dire < directions; dire++)
        {
            for(int g = 0; g < 4; g++)
            {
                hipMemcpyAsync(weight_ptr + offset,
                               B_ptr + (dire * 8 + 4 + gate_offset[g]) * hidden_size,
                               hidden_size * sizeof(DataType),
                               hipMemcpyDeviceToDevice,
                               ctx.get_stream().get());
                offset += hidden_size;
            }
        }
    }
}

shape miopen_lstm::compute_shape(const std::vector<shape>& inputs) const
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
argument miopen_lstm::compute(context& ctx,
                              const shape& output_shape,
                              const std::vector<argument>& args) const
{
    int directions = static_cast<int>(op.direction) >= 2 ? 2 : 1;
    if(directions == 2)
    {
        if(args[0].get_shape().type() == shape::float_type)
        {
            // 获取属性
            int sequence_len = args[0].get_shape().lens()[0]; // 输入数据的序列长度
            int batchsize    = args[0].get_shape().lens()[1]; // batchsize
            int input_size   = args[0].get_shape().lens()[2]; // 输入数据的特征维度
            int hidden_size  = args[2].get_shape().lens()[2]; // 隐藏层特征维度
            miopenDataType_t dataType = miopenFloat;

            // 创建x_desc,y_desc. 注意：个数与序列长度相等
            std::vector<miopenTensorDescriptor_t> x_desc(sequence_len);
            std::vector<miopenTensorDescriptor_t> y_desc(sequence_len);
            std::vector<int> lens(3);
            std::vector<int> strides(3);
            for(int i = 0; i < sequence_len; ++i)
            {
                // create x_desc
                miopenCreateTensorDescriptor(&(x_desc[i]));
                lens[0]    = batchsize;
                lens[1]    = input_size;
                lens[2]    = 1;
                strides[0] = lens[2] * lens[1];
                strides[1] = lens[2];
                strides[2] = 1;
                miopenSetTensorDescriptor(
                    x_desc[i], dataType, lens.size(), lens.data(), strides.data());

                // create y_desc
                miopenCreateTensorDescriptor(&(y_desc[i]));
                lens[0]    = batchsize;
                lens[1]    = hidden_size * (static_cast<int>(op.direction) >= 2 ? 2 : 1);
                lens[2]    = 1;
                strides[0] = lens[2] * lens[1];
                strides[1] = lens[2];
                strides[2] = 1;
                miopenSetTensorDescriptor(
                    y_desc[i], dataType, lens.size(), lens.data(), strides.data());
            }

            // 创建hx_desc,cx_desc,hy_desc,cy_desc
            auto hx_desc = make_tensor(args[4].get_shape());
            auto cx_desc = make_tensor(args[4].get_shape());
            auto hy_desc = make_tensor(args[4].get_shape());
            auto cy_desc = make_tensor(args[4].get_shape());

            // 创建weight_MIOpen
            if(is_first_copy_weight)
            {
                /* 将onnx中的权值转换为MIOpen格式

                onnx中权值的顺序为：IOFC (I表示input gate，O表示output gate，F表示forget
                gate，C表示cell gate) MIOpen的权值顺序为：IFOC
                所以需要进行IOFC->IFOC的转换，gate_offset表示他们的对应关系

                MIOpen的权值数据布局为：[w],[r],[wb],[rb]，如果是bidirectional的，则是[w1,w2],[r1,r2],[wb1,wb2],[rb1,rb2]
                */
                convert_onnx_to_MIOpen(ctx,
                                       args[1],
                                       args[2],
                                       has_bias ? args[3] : argument{},
                                       has_bias,
                                       static_cast<int>(op.direction) >= 2 ? 2 : 1,
                                       hidden_size,
                                       input_size,
                                       (float*)weight_MIOpen.data());
                is_first_copy_weight = false;
            }

            // 创建workspace
            size_t workspace_size;
            miopenGetRNNWorkspaceSize(ctx.get_stream().get_miopen(),
                                      rnn_desc.get(),
                                      sequence_len,
                                      x_desc.data(),
                                      &workspace_size);
            shape workspace_shape{shape::int8_type, {workspace_size}};
            static argument workspace = allocate_gpu(workspace_shape);
            if(workspace_shape.elements() > workspace.get_shape().elements())
            {
                workspace = allocate_gpu(workspace_shape);
            }

            // RNN推理（详细参数含义参考MIOpen文档）
            auto status =
                miopenRNNForwardInference(ctx.get_stream().get_miopen(),
                                          rnn_desc.get(),
                                          sequence_len,
                                          x_desc.data(),
                                          args[0].data(),
                                          hx_desc.get(),
                                          args.size() >= 6 ? args[4].data() : nullptr,
                                          cx_desc.get(),
                                          args.size() >= 6 ? args[5].data() : nullptr,
                                          w_desc.get(),
                                          weight_MIOpen.data(),
                                          y_desc.data(),
                                          args[args.size() - 1].data(),
                                          hy_desc.get(),
                                          nullptr, // 最后隐藏层的输出(the hidden layer output
                                                   // tensor),由于这里没有使用到，默认设置为NULL
                                          cy_desc.get(),
                                          nullptr, // 最后单元状态的输出(the cell layer output
                                                   // tensor),由于这里没有使用到，默认设置为NULL
                                          workspace.data(),
                                          workspace_size);
            if(status != miopenStatusSuccess)
                MIGRAPHX_THROW("MIOpen LSTM: running LSTM failed");

            // 释放资源
            for(int i = 0; i < sequence_len; ++i)
            {
                miopenDestroyTensorDescriptor(x_desc[i]);
                miopenDestroyTensorDescriptor(y_desc[i]);
            }

            // 由于ONNX的输出Tensor的layout为[seq_length, num_directions, batch_size, hidden_size]
            // 而MIOpen的输出Tensor的layout为[seq_length, batch_size, num_directions,
            // hidden_size],所以需要进行转置
            shape s          = args[args.size() - 1].get_shape();
            argument output1 = args[args.size() - 1].reshape(
                shape{s.type(), {s.lens()[0], s.lens()[2], s.lens()[1], s.lens()[3]}});
            shape s1         = output1.get_shape();
            argument output2 = output1.reshape(
                shape{s1.type(),
                      {s1.lens()[0], s1.lens()[2], s1.lens()[1], s1.lens()[3]},
                      {s1.strides()[0], s1.strides()[2], s1.strides()[1], s1.strides()[3]}});

            // 执行contiguous
            argument result2 = result.reshape(s);
            device::contiguous(ctx.get_stream().get(), result2, output2);

            return result2;
        }
        else if(args[0].get_shape().type() == shape::half_type)
        {
            // 获取属性
            int sequence_len = args[0].get_shape().lens()[0]; // 输入数据的序列长度
            int batchsize    = args[0].get_shape().lens()[1]; // batchsize
            int input_size   = args[0].get_shape().lens()[2]; // 输入数据的特征维度
            int hidden_size  = args[2].get_shape().lens()[2]; // 隐藏层特征维度
            miopenDataType_t dataType = miopenHalf;

            // 创建x_desc,y_desc. 注意：个数与序列长度相等
            std::vector<miopenTensorDescriptor_t> x_desc(sequence_len);
            std::vector<miopenTensorDescriptor_t> y_desc(sequence_len);
            std::vector<int> lens(3);
            std::vector<int> strides(3);
            for(int i = 0; i < sequence_len; ++i)
            {
                // create x_desc
                miopenCreateTensorDescriptor(&(x_desc[i]));
                lens[0]    = batchsize;
                lens[1]    = input_size;
                lens[2]    = 1;
                strides[0] = lens[2] * lens[1];
                strides[1] = lens[2];
                strides[2] = 1;
                miopenSetTensorDescriptor(
                    x_desc[i], dataType, lens.size(), lens.data(), strides.data());

                // create y_desc
                miopenCreateTensorDescriptor(&(y_desc[i]));
                lens[0]    = batchsize;
                lens[1]    = hidden_size * (static_cast<int>(op.direction) >= 2 ? 2 : 1);
                lens[2]    = 1;
                strides[0] = lens[2] * lens[1];
                strides[1] = lens[2];
                strides[2] = 1;
                miopenSetTensorDescriptor(
                    y_desc[i], dataType, lens.size(), lens.data(), strides.data());
            }

            // 创建hx_desc,cx_desc,hy_desc,cy_desc
            auto hx_desc = make_tensor(args[4].get_shape());
            auto cx_desc = make_tensor(args[4].get_shape());
            auto hy_desc = make_tensor(args[4].get_shape());
            auto cy_desc = make_tensor(args[4].get_shape());

            // 创建weight_MIOpen
            if(is_first_copy_weight)
            {
                /* 将onnx中的权值转换为MIOpen格式

                onnx中权值的顺序为：IOFC (I表示input gate，O表示output gate，F表示forget
                gate，C表示cell gate) MIOpen的权值顺序为：IFOC
                所以需要进行IOFC->IFOC的转换，gate_offset表示他们的对应关系

                MIOpen的权值数据布局为：[w],[r],[wb],[rb]，如果是bidirectional的，则是[w1,w2],[r1,r2],[wb1,wb2],[rb1,rb2]
                */
                convert_onnx_to_MIOpen(ctx,
                                       args[1],
                                       args[2],
                                       has_bias ? args[3] : argument{},
                                       has_bias,
                                       static_cast<int>(op.direction) >= 2 ? 2 : 1,
                                       hidden_size,
                                       input_size,
                                       (__fp16*)weight_MIOpen.data());
                is_first_copy_weight = false;
            }

            // 创建workspace
            size_t workspace_size;
            miopenGetRNNWorkspaceSize(ctx.get_stream().get_miopen(),
                                      rnn_desc.get(),
                                      sequence_len,
                                      x_desc.data(),
                                      &workspace_size);
            shape workspace_shape{shape::int8_type, {workspace_size}};
            static argument workspace = allocate_gpu(workspace_shape);
            if(workspace_shape.elements() > workspace.get_shape().elements())
            {
                workspace = allocate_gpu(workspace_shape);
            }

            // RNN推理（详细参数含义参考MIOpen文档）
            auto status =
                miopenRNNForwardInference(ctx.get_stream().get_miopen(),
                                          rnn_desc.get(),
                                          sequence_len,
                                          x_desc.data(),
                                          args[0].data(),
                                          hx_desc.get(),
                                          args.size() >= 6 ? args[4].data() : nullptr,
                                          cx_desc.get(),
                                          args.size() >= 6 ? args[5].data() : nullptr,
                                          w_desc.get(),
                                          weight_MIOpen.data(),
                                          y_desc.data(),
                                          args[args.size() - 1].data(),
                                          hy_desc.get(),
                                          nullptr, // 最后隐藏层的输出(the hidden layer output
                                                   // tensor),由于这里没有使用到，默认设置为NULL
                                          cy_desc.get(),
                                          nullptr, // 最后单元状态的输出(the cell layer output
                                                   // tensor),由于这里没有使用到，默认设置为NULL
                                          workspace.data(),
                                          workspace_size);
            if(status != miopenStatusSuccess)
                MIGRAPHX_THROW("MIOpen LSTM: running LSTM failed");

            // 释放资源
            for(int i = 0; i < sequence_len; ++i)
            {
                miopenDestroyTensorDescriptor(x_desc[i]);
                miopenDestroyTensorDescriptor(y_desc[i]);
            }

            // 由于ONNX的输出Tensor的layout为[seq_length, num_directions, batch_size, hidden_size]
            // 而MIOpen的输出Tensor的layout为[seq_length, batch_size, num_directions,
            // hidden_size],所以需要进行转置
            shape s          = args[args.size() - 1].get_shape();
            argument output1 = args[args.size() - 1].reshape(
                shape{s.type(), {s.lens()[0], s.lens()[2], s.lens()[1], s.lens()[3]}});
            shape s1         = output1.get_shape();
            argument output2 = output1.reshape(
                shape{s1.type(),
                      {s1.lens()[0], s1.lens()[2], s1.lens()[1], s1.lens()[3]},
                      {s1.strides()[0], s1.strides()[2], s1.strides()[1], s1.strides()[3]}});

            // 执行contiguous
            argument result2 = result.reshape(s);
            device::contiguous(ctx.get_stream().get(), result2, output2);

            return result2;
        }
    }
    else
    {
        if(args[0].get_shape().type() == shape::half_type)
        {
            // 获取属性
            int sequence_len = args[0].get_shape().lens()[0]; // 输入数据的序列长度
            int batchsize    = args[0].get_shape().lens()[1]; // batchsize
            int input_size   = args[0].get_shape().lens()[2]; // 输入数据的特征维度
            int hidden_size  = args[2].get_shape().lens()[2]; // 隐藏层特征维度
            miopenDataType_t dataType = miopenHalf;

            // 创建x_desc,y_desc. 注意：个数与序列长度相等
            std::vector<miopenTensorDescriptor_t> x_desc(sequence_len);
            std::vector<miopenTensorDescriptor_t> y_desc(sequence_len);
            std::vector<int> lens(3);
            std::vector<int> strides(3);
            for(int i = 0; i < sequence_len; ++i)
            {
                // create x_desc
                miopenCreateTensorDescriptor(&(x_desc[i]));
                lens[0]    = batchsize;
                lens[1]    = input_size;
                lens[2]    = 1;
                strides[0] = lens[2] * lens[1];
                strides[1] = lens[2];
                strides[2] = 1;
                miopenSetTensorDescriptor(
                    x_desc[i], dataType, lens.size(), lens.data(), strides.data());

                // create y_desc
                miopenCreateTensorDescriptor(&(y_desc[i]));
                lens[0]    = batchsize;
                lens[1]    = hidden_size * (static_cast<int>(op.direction) >= 2 ? 2 : 1);
                lens[2]    = 1;
                strides[0] = lens[2] * lens[1];
                strides[1] = lens[2];
                strides[2] = 1;
                miopenSetTensorDescriptor(
                    y_desc[i], dataType, lens.size(), lens.data(), strides.data());
            }

            // 创建hx_desc,cx_desc,hy_desc,cy_desc
            auto hx_desc = make_tensor(args[4].get_shape());
            auto cx_desc = make_tensor(args[4].get_shape());
            auto hy_desc = make_tensor(args[4].get_shape());
            auto cy_desc = make_tensor(args[4].get_shape());

            // 创建weight_MIOpen
            if(is_first_copy_weight)
            {
                /* 将onnx中的权值转换为MIOpen格式

                onnx中权值的顺序为：IOFC (I表示input gate，O表示output gate，F表示forget
                gate，C表示cell gate) MIOpen的权值顺序为：IFOC
                所以需要进行IOFC->IFOC的转换，gate_offset表示他们的对应关系

                MIOpen的权值数据布局为：[w],[r],[wb],[rb]，如果是bidirectional的，则是[w1,w2],[r1,r2],[wb1,wb2],[rb1,rb2]
                */
                convert_onnx_to_MIOpen(ctx,
                                       args[1],
                                       args[2],
                                       has_bias ? args[3] : argument{},
                                       has_bias,
                                       static_cast<int>(op.direction) >= 2 ? 2 : 1,
                                       hidden_size,
                                       input_size,
                                       (__fp16*)weight_MIOpen.data());
                is_first_copy_weight = false;
            }

            // 创建workspace
            size_t workspace_size;
            miopenGetRNNWorkspaceSize(ctx.get_stream().get_miopen(),
                                      rnn_desc.get(),
                                      sequence_len,
                                      x_desc.data(),
                                      &workspace_size);
            shape workspace_shape{shape::int8_type, {workspace_size}};
            static argument workspace = allocate_gpu(workspace_shape);
            if(workspace_shape.elements() > workspace.get_shape().elements())
            {
                workspace = allocate_gpu(workspace_shape);
            }

            // RNN推理（详细参数含义参考MIOpen文档）
            auto status =
                miopenRNNForwardInference(ctx.get_stream().get_miopen(),
                                          rnn_desc.get(),
                                          sequence_len,
                                          x_desc.data(),
                                          args[0].data(),
                                          hx_desc.get(),
                                          args.size() >= 6 ? args[4].data() : nullptr,
                                          cx_desc.get(),
                                          args.size() >= 6 ? args[5].data() : nullptr,
                                          w_desc.get(),
                                          weight_MIOpen.data(),
                                          y_desc.data(),
                                          args[args.size() - 1].data(),
                                          hy_desc.get(),
                                          nullptr, // 最后隐藏层的输出(the hidden layer output
                                                   // tensor),由于这里没有使用到，默认设置为NULL
                                          cy_desc.get(),
                                          nullptr, // 最后单元状态的输出(the cell layer output
                                                   // tensor),由于这里没有使用到，默认设置为NULL
                                          workspace.data(),
                                          workspace_size);
            if(status != miopenStatusSuccess)
                MIGRAPHX_THROW("MIOpen LSTM: running LSTM failed");

            // 释放资源
            for(int i = 0; i < sequence_len; ++i)
            {
                miopenDestroyTensorDescriptor(x_desc[i]);
                miopenDestroyTensorDescriptor(y_desc[i]);
            }
            return args[args.size() - 1];
        }
        else if(args[0].get_shape().type() == shape::float_type)
        {
            // 获取属性
            int sequence_len = args[0].get_shape().lens()[0]; // 输入数据的序列长度
            int batchsize    = args[0].get_shape().lens()[1]; // batchsize
            int input_size   = args[0].get_shape().lens()[2]; // 输入数据的特征维度
            int hidden_size  = args[2].get_shape().lens()[2]; // 隐藏层特征维度
            miopenDataType_t dataType = miopenFloat;

            // 创建x_desc,y_desc. 注意：个数与序列长度相等
            std::vector<miopenTensorDescriptor_t> x_desc(sequence_len);
            std::vector<miopenTensorDescriptor_t> y_desc(sequence_len);
            std::vector<int> lens(3);
            std::vector<int> strides(3);
            for(int i = 0; i < sequence_len; ++i)
            {
                // create x_desc
                miopenCreateTensorDescriptor(&(x_desc[i]));
                lens[0]    = batchsize;
                lens[1]    = input_size;
                lens[2]    = 1;
                strides[0] = lens[2] * lens[1];
                strides[1] = lens[2];
                strides[2] = 1;
                miopenSetTensorDescriptor(
                    x_desc[i], dataType, lens.size(), lens.data(), strides.data());

                // create y_desc
                miopenCreateTensorDescriptor(&(y_desc[i]));
                lens[0]    = batchsize;
                lens[1]    = hidden_size * (static_cast<int>(op.direction) >= 2 ? 2 : 1);
                lens[2]    = 1;
                strides[0] = lens[2] * lens[1];
                strides[1] = lens[2];
                strides[2] = 1;
                miopenSetTensorDescriptor(
                    y_desc[i], dataType, lens.size(), lens.data(), strides.data());
            }

            // 创建hx_desc,cx_desc,hy_desc,cy_desc
            auto hx_desc = make_tensor(args[4].get_shape());
            auto cx_desc = make_tensor(args[4].get_shape());
            auto hy_desc = make_tensor(args[4].get_shape());
            auto cy_desc = make_tensor(args[4].get_shape());

            // 创建weight_MIOpen
            if(is_first_copy_weight)
            {
                /* 将onnx中的权值转换为MIOpen格式

                onnx中权值的顺序为：IOFC (I表示input gate，O表示output gate，F表示forget
                gate，C表示cell gate) MIOpen的权值顺序为：IFOC
                所以需要进行IOFC->IFOC的转换，gate_offset表示他们的对应关系

                MIOpen的权值数据布局为：[w],[r],[wb],[rb]，如果是bidirectional的，则是[w1,w2],[r1,r2],[wb1,wb2],[rb1,rb2]
                */
                convert_onnx_to_MIOpen(ctx,
                                       args[1],
                                       args[2],
                                       has_bias ? args[3] : argument{},
                                       has_bias,
                                       static_cast<int>(op.direction) >= 2 ? 2 : 1,
                                       hidden_size,
                                       input_size,
                                       (float*)weight_MIOpen.data());
                is_first_copy_weight = false;
            }

            // 创建workspace
            size_t workspace_size;
            miopenGetRNNWorkspaceSize(ctx.get_stream().get_miopen(),
                                      rnn_desc.get(),
                                      sequence_len,
                                      x_desc.data(),
                                      &workspace_size);
            shape workspace_shape{shape::int8_type, {workspace_size}};
            static argument workspace = allocate_gpu(workspace_shape);
            if(workspace_shape.elements() > workspace.get_shape().elements())
            {
                workspace = allocate_gpu(workspace_shape);
            }

            // RNN推理（详细参数含义参考MIOpen文档）
            auto status =
                miopenRNNForwardInference(ctx.get_stream().get_miopen(),
                                          rnn_desc.get(),
                                          sequence_len,
                                          x_desc.data(),
                                          args[0].data(),
                                          hx_desc.get(),
                                          args.size() >= 6 ? args[4].data() : nullptr,
                                          cx_desc.get(),
                                          args.size() >= 6 ? args[5].data() : nullptr,
                                          w_desc.get(),
                                          weight_MIOpen.data(),
                                          y_desc.data(),
                                          args[args.size() - 1].data(),
                                          hy_desc.get(),
                                          nullptr, // 最后隐藏层的输出(the hidden layer output
                                                   // tensor),由于这里没有使用到，默认设置为NULL
                                          cy_desc.get(),
                                          nullptr, // 最后单元状态的输出(the cell layer output
                                                   // tensor),由于这里没有使用到，默认设置为NULL
                                          workspace.data(),
                                          workspace_size);
            if(status != miopenStatusSuccess)
                MIGRAPHX_THROW("MIOpen LSTM: running LSTM failed");

            // 释放资源
            for(int i = 0; i < sequence_len; ++i)
            {
                miopenDestroyTensorDescriptor(x_desc[i]);
                miopenDestroyTensorDescriptor(y_desc[i]);
            }
            return args[args.size() - 1];
        }
    }
}
void miopen_lstm::finalize(context& ctx, const shape& output_shape, std::vector<shape> inputs)
{
    if(is_first_finalize)
    {
        shape input  = inputs[0];
        shape output = inputs[inputs.size() - 1];
        result       = allocate_gpu(output);

        // 创建weight_MIOpen
        std::size_t dimW[3];
        if(has_bias)
        {
            dimW[0] = inputs[1].elements() + inputs[2].elements() + inputs[3].elements();
        }
        else
        {
            dimW[0] = inputs[1].elements() + inputs[2].elements();
        }
        dimW[1] = 1;
        dimW[2] = 1;
        shape weight_shape{inputs[1].type(), {dimW[0], dimW[1], dimW[2]}};
        if(w_desc == nullptr)
        {
            w_desc = make_tensor(weight_shape);
        }
        weight_MIOpen = allocate_gpu(weight_shape);

        is_first_finalize = false;
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
