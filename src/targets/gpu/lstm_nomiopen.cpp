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
#include <migraphx/generate.hpp>
#include <migraphx/SimpleLog.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static void LSTM_Single(const float *x, float *y, const float *w, const float *r, const float *b,
                          float *h_t, float *c_t,
                          const int T, const int batch_size, const int input_size, const int hidden_size, int reverse) {
    //num_directions = 1 for all below
    //X shape [sequence batch_size input_size]
    const int x_page_size = batch_size * input_size;
    
    //Y shape [sequence batch_size num_directions * hidden_size]
    const int y_page_size = batch_size * hidden_size;
    
    //W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    const int w_page_size = hidden_size * input_size;
    auto w_x_I = w;
    auto w_x_O = w_x_I + w_page_size;
    auto w_x_F = w_x_O + w_page_size;
    auto w_x_C = w_x_F + w_page_size;
    
    //R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    int r_page_size = hidden_size * hidden_size;
    auto r_x_I = r;
    auto r_x_O = r_x_I + r_page_size;
    auto r_x_F = r_x_O + r_page_size;
    auto r_x_C = r_x_F + r_page_size;
    
    //B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    int b_page_size = hidden_size;
    auto b_w_I = b;
    auto b_w_O = b_w_I + b_page_size;
    auto b_w_F = b_w_O + b_page_size;
    auto b_w_C = b_w_F + b_page_size;
    
    auto b_r_I = b_w_C + b_page_size;
    auto b_r_O = b_r_I + b_page_size;
    auto b_r_F = b_r_O + b_page_size;
    auto b_r_C = b_r_F + b_page_size;
    
    //temp gates, shape [hidden_size, 4]
    auto gates = std::shared_ptr<float>(new float[hidden_size * 4], [](float* p) { delete[] p; });
    
    for (int t = 0; t < T; t++) {
        int ti = reverse ? T - 1 - t : t;

        const float* x_t = x + ti * x_page_size;
        float* y_t = y + ti *y_page_size;
        
        for (int b = 0; b < batch_size; b++) {
            const float* x_t_b = x_t + b * input_size;
            float* h_t_b = h_t + b * hidden_size;
            float* c_t_b = c_t + b * hidden_size;
            //float*gates_b = (float *)gates.get() + b * output_size * 4;
            
            for (int q = 0; q < hidden_size; q++) {
                auto gates_data = (float *)gates.get() + q * 4;
                
                //W weights
                auto w_x_I_o = w_x_I + q * input_size;
                auto w_x_O_o = w_x_O + q * input_size;
                auto w_x_F_o = w_x_F + q * input_size;
                auto w_x_C_o = w_x_C + q * input_size;

                auto r_x_I_o = r_x_I + q * hidden_size;
                auto r_x_O_o = r_x_O + q * hidden_size;
                auto r_x_F_o = r_x_F + q * hidden_size;
                auto r_x_C_o = r_x_C + q * hidden_size;
                
                //bias
                float I = b_w_I[q] + b_r_I[q];
                float O = b_w_O[q] + b_r_O[q];
                float F = b_w_F[q] + b_r_F[q];
                float C = b_w_C[q] + b_r_C[q];

                for (int i = 0; i < input_size; i++) {
                    I += w_x_I_o[i] * x_t_b[i];
                    O += w_x_O_o[i] * x_t_b[i];
                    F += w_x_F_o[i] * x_t_b[i];
                    C += w_x_C_o[i] * x_t_b[i];
                }

                for (int i = 0; i < hidden_size; i++) {
                    I += r_x_I_o[i] * h_t_b[i];
                    O += r_x_O_o[i] * h_t_b[i];
                    F += r_x_F_o[i] * h_t_b[i];
                    C += r_x_C_o[i] * h_t_b[i];
                }

                gates_data[0] = I;
                gates_data[1] = O;
                gates_data[2] = F;
                gates_data[3] = C;
            }
            
            float* output_data = y_t + b *hidden_size;
            for (int q = 0; q < hidden_size; q++) {
                const auto gates_data = (float *)gates.get() + q * 4;

                float I = gates_data[0];
                float O = gates_data[1];
                float F = gates_data[2];
                float C = gates_data[3];

                I = 1.f / (1.f + exp(-I));
                F = 1.f / (1.f + exp(-F));
                O = 1.f / (1.f + exp(-O));
                C = tanh(C);

                float cell2 = F * c_t_b[q] + I * C;
                float H = O * tanh(cell2);
                c_t_b[q] = cell2;
                h_t_b[q] = H;
                output_data[q] = H;
            }
        }
    }
}

shape lstm_nomiopen::compute_shape(const std::vector<shape>& inputs) const
{ 
    std::vector<shape> inputs2;
    inputs2.push_back(inputs[0]);
    check_shapes{inputs2, *this}.standard();
    return op.compute_shape(inputs);
}

/*
args[0]:输入tensor
args[1]:input weight matrix
args[2]:hidden state weight matrix
args[3]:bias
args[5]:intial hidden state
args[6]:initial cell value
*/
argument lstm_nomiopen::compute(context& ctx,
                                     const shape& output_shape,
                                     const std::vector<argument>& args) const
{
    // 这里一定要加入同步，否则结果不对
    ctx.finish();

    // 获取属性
    int sequenceLen = args[0].get_shape().lens()[0];// 输入数据的序列长度
    int batchsize = args[0].get_shape().lens()[1];// batchsize
    int input_size=args[0].get_shape().lens()[2]; // 输入数据的特征维度
    int hidden_size = args[2].get_shape().lens()[2]; // 隐藏层特征维度
    int num_directions =static_cast<int>(op.direction) >= 2 ? 2 : 1;
    int T=sequenceLen;
    int batch=batchsize;

    // 将数据拷贝到CPU
    std::vector<argument> args2;
    for(int i=0;i<args.size();++i)
    {
        args2.push_back(from_gpu(args[i]));
    }

    //X shape [sequence batch_size input_size]
    float *x = (float *)(args2[0].data());
    
    //Y shape [sequence batch_size num_directions *hidden_size]
    float *y = (float *)(args2[args2.size()-1].data());
    
    //W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    float *w = (float *)(args2[1].data());
    
    //R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    float *r = (float *)(args2[2].data());
    
    //B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    float *b = (float *)(args2[3].data());

    //initial_h, initial value of the hidden, If not specified - assumed to be 0. shape [num_directions, batch_size, hidden_size]
    auto h_t = (float *)malloc(num_directions * batch * hidden_size * sizeof(float));
    if (args.size()>=6)
    {
        auto h_0 = args[5].data();
        memcpy((void *)h_t, h_0, num_directions * batch * hidden_size * sizeof(float));
    } 
    else 
    {
        memset(h_t, 0, num_directions * batch * hidden_size * sizeof(float));
    }
    
    //initial_c, initial value of the cell, If not specified - assumed to be 0. shape [num_directions, batch_size, hidden_size]
    auto c_t = (float *)malloc(num_directions * batch * hidden_size * sizeof(float));
    if (args.size()>=7)
    {
        auto c_0 = args[6].data();
        memcpy((void *)c_t, c_0, num_directions * batch * hidden_size * sizeof(float));
    } 
    else 
    {
        memset(c_t, 0, num_directions * batch * hidden_size * sizeof(float));
    }

    if (num_directions==1) 
    {
        LSTM_Single(x, y, w, r, b, h_t, c_t, T, batch, input_size, hidden_size, (int)op.direction);

    } 
    else//  num_directions == 2
    {
        //Y shape [num_directions sequence batch_size hidden_size]
        auto y_temp = std::shared_ptr<float>(new float[num_directions*T*batch*hidden_size], [](float* p) { delete[] p; });
        auto y0 = y_temp.get();
        auto y1 = y0 + T * batch * hidden_size;
        LSTM_Single(x, y0, w, r, b, h_t, c_t, T, batch, input_size, hidden_size, 0);
        
        auto w1 = w + 4*hidden_size*input_size;
        auto r1 = r + 4*hidden_size*hidden_size;
        auto b1 = b + 8*hidden_size;
        auto h_t1 = h_t + batch*hidden_size;
        auto c_t1 = c_t + batch*hidden_size;
        LSTM_Single(x, y1, w1, r1, b1, h_t1, c_t1, T, batch, input_size, hidden_size, 1);
        
        //transpose [num_directions sequence batch_size hidden_size] to [sequence batch_size num_directions*hidden_size]
        for (int i = 0; i < T*batch; i++) {
            auto y0_data = y0 + i*hidden_size;
            auto y1_data = y1 + i*hidden_size;
            auto y_data = y + i*num_directions*hidden_size;

            memcpy(y_data, y0_data, hidden_size * sizeof(float));
            memcpy(y_data + hidden_size, y1_data, hidden_size * sizeof(float));
        }
    }

    // 将输出数据拷贝到GPU
    migraphx::argument result_host=to_gpu(args2[args2.size()-1],true);
    for(int i=0;i<result_host.get_shape().elements();++i)
    {
        ((float *)args[args.size()-1].data())[i]=((float *)result_host.data())[i];
    }

    free(h_t);
    free(c_t);
    return args[args.size()-1];
}
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
