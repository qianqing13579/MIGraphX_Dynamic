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

#include <migraphx/gpu/device/lstm.hpp>
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/gemm_impl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T, index_int N>
using VecType = T __attribute__((ext_vector_type(N)));

template <typename DataType>
__device__ DataType sigmoid(DataType x)
{
    return DataType(1) / (DataType(1) + exp(-x));
}

template <typename DataType>
__global__ void Add(int nthreads,
                    DataType* HR,
                    DataType* XW,
                    DataType* wb,
                    DataType* rb,
                    int hidden_size,
                    int batch_size)
{
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, nthreads)
    {
        // 确定索引
        int index_w = index % (4 * hidden_size); // w维度索引
        int index_h = index / (4 * hidden_size) % batch_size;

        XW[index] += HR[index];
        XW[index] += (wb[index_w] + rb[index_w]);
    }
}

template <typename DataType>
__global__ void Activation(int nthreads, int hidden_size, DataType* XW)
{
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, nthreads)
    {
        // 计算在w维度的索引
        int index_w = index % (4 * hidden_size);
        XW[index]   = index_w < 3 * hidden_size ? sigmoid(XW[index]) : tanh(XW[index]);
    }
}

// vec4向量化
template <typename DataType>
__global__ void add_activation_vec(int nthreads,
                                   DataType* HR,
                                   DataType* XW,
                                   DataType* wb,
                                   DataType* rb,
                                   int hidden_size,
                                   int batch_size)
{
    using Vec4 = VecType<DataType, 4>;
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, nthreads)
    {
        // 获取线程对应的vec4元素
        Vec4 xw = ((Vec4*)XW)[index];
        Vec4 hr = ((Vec4*)HR)[index];

        // vec4元素的起始索引
        int index2 = index * 4;

#pragma unroll
        // 处理vec4中每个元素，注意要循环展开
        for(int i = 0; i < 4; ++i)
        {
            // vec4中每个元素索引
            int index3 = index2 + i;

            // w维度索引
            int index_w = index3 % (4 * hidden_size);
            int index_h = index3 / (4 * hidden_size) % batch_size;

            xw[i] += hr[i];
            xw[i] += (wb[index_w] + rb[index_w]);

            // 计算激活值
            xw[i] = index_w < 3 * hidden_size ? sigmoid(xw[i]) : tanh(xw[i]);
        }

        ((Vec4*)XW)[index] = xw;
    }
}

template <typename DataType>
__global__ void LSTMOutput(int nthreads,
                           int hidden_size,
                           DataType* XW_t,
                           DataType* h_t_1,
                           DataType* c_t_1,
                           DataType* h_t,
                           DataType* c_t,
                           DataType* y)
{
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, nthreads)
    {
        int index_h = index / hidden_size; // batch索引
        int index_w = index % hidden_size; // 隐藏层哪一个神经元

        DataType* XW_t_offset = XW_t + index_h * (4 * hidden_size);
        DataType i_t          = XW_t_offset[index_w];
        DataType o_t          = XW_t_offset[index_w + hidden_size];
        DataType f_t          = XW_t_offset[index_w + 2 * hidden_size];
        DataType g_t          = XW_t_offset[index_w + 3 * hidden_size];

        DataType* c_t_1_offset = c_t_1 + index_h * hidden_size;
        DataType* h_t_1_offset = h_t_1 + index_h * hidden_size;
        DataType* c_t_offset   = c_t + index_h * hidden_size;
        DataType* h_t_offset   = h_t + index_h * hidden_size;
        DataType* y_t_offset   = y + index_h * hidden_size;

        DataType cell       = f_t * c_t_1_offset[index_w] + i_t * g_t;
        DataType hidden     = o_t * tanh(cell);
        h_t_offset[index_w] = hidden;
        c_t_offset[index_w] = cell;
        y_t_offset[index_w] = hidden;
    }
}

template <typename DataType>
void LSTM_Single(context& ctx,
                 DataType* x,
                 DataType* y, // 输入，输出
                 DataType* w,
                 DataType* r,
                 DataType* b, // 权重
                 DataType* h_0,
                 DataType* c_0,
                 DataType* h_t,
                 DataType* c_t,
                 DataType* XW_,
                 DataType* x_reverse_,
                 DataType* HR_,
                 int T,
                 int batch_size,
                 int input_size,
                 int hidden_size,
                 shape::type_t data_type,
                 int reverse)
{

    // 计算x*w
    argument X{shape{data_type, {(std::size_t)T * batch_size, (std::size_t)input_size}}, x};
    argument W{shape{data_type, {(std::size_t)input_size, (std::size_t)4 * hidden_size}}, w};
    argument XW{shape{data_type, {(std::size_t)T * batch_size, (std::size_t)4 * hidden_size}}, XW_};
    float alpha = 1.0;
    float beta  = 0.0;
    if(reverse == 0)
    {
        gemm(ctx, XW.get_shape(), {X, W, XW}, alpha, beta, true, false);
    }
    else // 如果是双向LSTM，对X做反向
    {
        argument x_reverse{shape{data_type, {(std::size_t)T * batch_size, (std::size_t)input_size}},
                           x_reverse_};
        for(int t = 0; t < T; ++t)
        {
            DataType* x_reverse_data = (DataType*)x_reverse.data();
            DataType* dst            = x_reverse_data + (T - 1 - t) * batch_size * input_size;
            hipMemcpyAsync(dst,
                           x + t * batch_size * input_size,
                           batch_size * input_size * sizeof(DataType),
                           hipMemcpyDeviceToDevice,
                           ctx.get_stream().get());
        }
        gemm(ctx, XW.get_shape(), {x_reverse, W, XW}, alpha, beta, true, false);
    }

    for(int t = 0; t < T; ++t)
    {
        int ti        = reverse ? T - 1 - t : t;
        DataType* y_t = y + ti * batch_size *
                                hidden_size; // y: [num_directions sequence batch_size hidden_size]
        DataType* h_t_1;
        DataType* c_t_1;
        if(t == 0)
        {
            h_t_1 = h_0;
            c_t_1 = c_0;
        }
        else
        {
            h_t_1 = h_t;
            c_t_1 = c_t;
        }
        // 计算h*r
        argument H{shape{data_type, {(std::size_t)batch_size, (std::size_t)hidden_size}}, h_t_1};
        argument R{shape{data_type, {(std::size_t)hidden_size, (std::size_t)4 * hidden_size}}, r};
        argument HR{shape{data_type, {(std::size_t)batch_size, (std::size_t)4 * hidden_size}}, HR_};
        float alpha = 1.0;
        float beta  = 0.0;
        gemm(ctx, HR.get_shape(), {H, R, HR}, alpha, beta, true, false);

        // 计算x*w+h*r+b
        DataType* XW_P = (DataType*)XW.data();
        DataType* XW_t = XW_P + t * (4 * hidden_size * batch_size);
        DataType* HR_t = (DataType*)HR.data();
        add_activation_vec<<<get_number_blocks(batch_size * hidden_size),
                             NUM_THREADS_PER_BLOCK,
                             0,
                             ctx.get_stream().get()>>>(
            batch_size * hidden_size, HR_t, XW_t, b, b + 4 * hidden_size, hidden_size, batch_size);

        // 计算隐藏层的输出
        LSTMOutput<<<get_number_blocks(batch_size * hidden_size),
                     NUM_THREADS_PER_BLOCK,
                     0,
                     ctx.get_stream().get()>>>(
            batch_size * hidden_size, hidden_size, XW_t, h_t_1, c_t_1, h_t, c_t, y_t);
    }
}

void LSTM(context& ctx,
          const argument& Y,
          const argument& X,
          const argument& W,
          const argument& R,
          const argument& B,
          const argument& h_0,
          const argument& c_0,
          const argument& h_t,
          const argument& c_t,
          const argument& XW,
          const argument& x_reverse,
          const argument& HR,
          const argument& result_gpu,
          int num_directions)
{
    // 获取属性
    int sequenceLen = X.get_shape().lens()[0]; // 输入数据的序列长度
    int batchsize   = X.get_shape().lens()[1]; // batchsize
    int input_size  = X.get_shape().lens()[2]; // 输入数据的特征维度
    int hidden_size =
        R.get_shape()
            .lens()[1]; // 隐藏层特征维度,注意，由于这里R被转置了，所以lens()[1]表示hidden_size
    int T     = sequenceLen;
    int batch = batchsize;

    if(X.get_shape().type() == shape::float_type)
    {
        // X :[sequence batch_size input_size]
        float* x = (float*)(X.data());

        // Y: [seq_length, num_directions, batch_size, hidden_size]
        float* y = (float*)(Y.data());

        // W[iofc]:[num_directions, 4*hidden_size, input_size]
        float* w = (float*)(W.data());

        // R[iofc]:[num_directions, 4*hidden_size, hidden_size]
        float* r = (float*)(R.data());

        // B(Wb[iofc], Rb[iofc]):[num_directions, 8*hidden_size]
        float* b = (float*)(B.data());

        if(num_directions == 1)
        {
            LSTM_Single(ctx,
                        x,
                        y,
                        w,
                        r,
                        b,
                        (float*)h_0.data(),
                        (float*)c_0.data(),
                        (float*)h_t.data(),
                        (float*)c_t.data(),
                        (float*)XW.data(),
                        (float*)x_reverse.data(),
                        (float*)HR.data(),
                        T,
                        batch,
                        input_size,
                        hidden_size,
                        shape::float_type,
                        0);
        }
        else //  num_directions == 2
        {
            argument y_temp = result_gpu.reshape(
                shape{shape::float_type, {(std::size_t)num_directions * T * batch * hidden_size}});
            float* y0 = (float*)y_temp.data();
            float* y1 = y0 + T * batch * hidden_size;
            LSTM_Single(ctx,
                        x,
                        y0,
                        w,
                        r,
                        b,
                        (float*)h_0.data(),
                        (float*)c_0.data(),
                        (float*)h_t.data(),
                        (float*)c_t.data(),
                        (float*)XW.data(),
                        (float*)x_reverse.data(),
                        (float*)HR.data(),
                        T,
                        batch,
                        input_size,
                        hidden_size,
                        shape::float_type,
                        0);

            float* w1    = w + 4 * hidden_size * input_size;
            float* r1    = r + 4 * hidden_size * hidden_size;
            float* b1    = b + 8 * hidden_size;
            float* h_0_p = (float*)h_0.data();
            float* h_01  = h_0_p + batch * hidden_size;
            float* c_0_p = (float*)c_0.data();
            float* c_01  = c_0_p + batch * hidden_size;
            LSTM_Single(ctx,
                        x,
                        y1,
                        w1,
                        r1,
                        b1,
                        h_01,
                        c_01,
                        (float*)h_t.data(),
                        (float*)c_t.data(),
                        (float*)XW.data(),
                        (float*)x_reverse.data(),
                        (float*)HR.data(),
                        T,
                        batch,
                        input_size,
                        hidden_size,
                        shape::float_type,
                        1);

            // [num_directions sequence batch_size hidden_size] -> [sequence, num_directions,
            // batch_size, hidden_size]
            argument output1 = y_temp.reshape(shape{shape::float_type,
                                                    {(std::size_t)num_directions,
                                                     (std::size_t)T,
                                                     (std::size_t)batch,
                                                     (std::size_t)hidden_size}});
            shape s1         = output1.get_shape();
            argument output2 = output1.reshape(
                shape{s1.type(),
                      {s1.lens()[1], s1.lens()[0], s1.lens()[2], s1.lens()[3]},
                      {s1.strides()[1], s1.strides()[0], s1.strides()[2], s1.strides()[3]}});
            contiguous(ctx.get_stream().get(), Y, output2);
        }
    }
    else if(X.get_shape().type() == shape::half_type)
    {

        // X :[sequence batch_size input_size]
        __fp16* x = (__fp16*)(X.data());

        // Y: [seq_length, num_directions, batch_size, hidden_size]
        __fp16* y = (__fp16*)(Y.data());

        // W[iofc]:[num_directions, 4*hidden_size, input_size]
        __fp16* w = (__fp16*)(W.data());

        // R[iofc]:[num_directions, 4*hidden_size, hidden_size]
        __fp16* r = (__fp16*)(R.data());

        // B(Wb[iofc], Rb[iofc]):[num_directions, 8*hidden_size]
        __fp16* b = (__fp16*)(B.data());

        if(num_directions == 1)
        {
            LSTM_Single(ctx,
                        x,
                        y,
                        w,
                        r,
                        b,
                        (__fp16*)h_0.data(),
                        (__fp16*)c_0.data(),
                        (__fp16*)h_t.data(),
                        (__fp16*)c_t.data(),
                        (__fp16*)XW.data(),
                        (__fp16*)x_reverse.data(),
                        (__fp16*)HR.data(),
                        T,
                        batch,
                        input_size,
                        hidden_size,
                        shape::half_type,
                        0);
        }
        else //  num_directions == 2
        {
            argument y_temp = result_gpu.reshape(
                shape{shape::half_type, {(std::size_t)num_directions * T * batch * hidden_size}});
            __fp16* y0 = (__fp16*)y_temp.data();
            __fp16* y1 = y0 + T * batch * hidden_size;
            LSTM_Single(ctx,
                        x,
                        y0,
                        w,
                        r,
                        b,
                        (__fp16*)h_0.data(),
                        (__fp16*)c_0.data(),
                        (__fp16*)h_t.data(),
                        (__fp16*)c_t.data(),
                        (__fp16*)XW.data(),
                        (__fp16*)x_reverse.data(),
                        (__fp16*)HR.data(),
                        T,
                        batch,
                        input_size,
                        hidden_size,
                        shape::half_type,
                        0);

            __fp16* w1    = w + 4 * hidden_size * input_size;
            __fp16* r1    = r + 4 * hidden_size * hidden_size;
            __fp16* b1    = b + 8 * hidden_size;
            __fp16* h_0_p = (__fp16*)h_0.data();
            __fp16* h_01  = h_0_p + batch * hidden_size;
            __fp16* c_0_p = (__fp16*)c_0.data();
            __fp16* c_01  = c_0_p + batch * hidden_size;
            LSTM_Single(ctx,
                        x,
                        y1,
                        w1,
                        r1,
                        b1,
                        h_01,
                        c_01,
                        (__fp16*)h_t.data(),
                        (__fp16*)c_t.data(),
                        (__fp16*)XW.data(),
                        (__fp16*)x_reverse.data(),
                        (__fp16*)HR.data(),
                        T,
                        batch,
                        input_size,
                        hidden_size,
                        shape::half_type,
                        1);

            // [num_directions sequence batch_size hidden_size] -> [sequence, num_directions,
            // batch_size, hidden_size]
            argument output1 = y_temp.reshape(shape{shape::half_type,
                                                    {(std::size_t)num_directions,
                                                     (std::size_t)T,
                                                     (std::size_t)batch,
                                                     (std::size_t)hidden_size}});
            shape s1         = output1.get_shape();
            argument output2 = output1.reshape(
                shape{s1.type(),
                      {s1.lens()[1], s1.lens()[0], s1.lens()[2], s1.lens()[3]},
                      {s1.strides()[1], s1.strides()[0], s1.strides()[2], s1.strides()[3]}});
            contiguous(ctx.get_stream().get(), Y, output2);
        }
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
