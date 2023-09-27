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
#include <migraphx/gpu/convolution_dynamic.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/device/convolution_2d_fp32.hpp>
#include <migraphx/gpu/device/convolution_2d_fp16.hpp>


namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_convolution_dynamic::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(4).standard();
    std::vector<shape> conv_inputs(inputs.begin(), inputs.begin() + 2);
    check_shapes{conv_inputs, *this}.max_ndims(5);
    return op.normalize_compute_shape(conv_inputs);
}

inline shape reshape_if_1d(const shape& input)
{
    shape new_shape{input};
    auto dims = new_shape.lens();

    if(dims.size() == 3)
    {
        std::vector<size_t> new_dims = dims;
        new_dims.insert(new_dims.begin() + 2, 1);
        new_shape = shape{input.type(), new_dims};
    }
    return new_shape;
}

argument miopen_convolution_dynamic::compute(context& ctx,
                                     const shape& output_shape,
                                     const std::vector<argument>& args) const
{
    shape kernel_shape=args[1].get_shape();
    
    bool is_conv_1d=(kernel_shape.lens().size()==3&&op.padding.size()==2&&op.padding[0]==op.padding[1]&&op.padding[0]==0); // 如果是1维卷积且pad是对称的
    bool is_conv_2d=(kernel_shape.lens().size()==4&&op.padding.size()==4&&op.padding[0]==op.padding[2]&&op.padding[1]==op.padding[3]&&op.padding[0]==0&&op.padding[1]==0);
    
    
    // is_conv_1d或者is_conv_2d的FP32情况下使用igemm的实现，可以解决启动性能问题
    if(kernel_shape.type()==shape::float_type&&(is_conv_1d||is_conv_2d))
    {
        if(is_conv_1d)
        {
            std::vector<std::size_t> new_padding(4,0);
            new_padding[0]=op.padding[0];
            new_padding[2]=op.padding[1];

            std::vector<std::size_t> new_stride(2,1);
            new_stride[1]=op.stride[0];

            std::vector<std::size_t> new_dilation(2,1);
            new_dilation[1]=op.dilation[0];

            std::vector<std::size_t> new_x_lens(4,1);
            new_x_lens[0]=args[0].get_shape().lens()[0];
            new_x_lens[1]=args[0].get_shape().lens()[1];
            new_x_lens[3]=args[0].get_shape().lens()[2];

            std::vector<std::size_t> new_w_lens(4,1);
            new_w_lens[0]=args[1].get_shape().lens()[0];
            new_w_lens[1]=args[1].get_shape().lens()[1];
            new_w_lens[3]=args[1].get_shape().lens()[2];

            // 使用2维计算
            device::convolution_2d_fp32(ctx.get_stream().get(), 
                                    args[3], 
                                    args[0].reshape(shape{args[0].get_shape().type(),new_x_lens}),
                                    args[1].reshape(shape{args[1].get_shape().type(),new_w_lens}),
                                    new_padding,new_stride,new_dilation,op.group);
        }
        else if(is_conv_2d)
        {
            // fp32
            device::convolution_2d_fp32(ctx.get_stream().get(), args[3], args[0],args[1],op.padding,op.stride,op.dilation,op.group);
        }

    }
    // 其他情况使用MIOpen
    else
    {
        auto x_desc = make_tensor(reshape_if_1d(args[0].get_shape()));
        auto w_desc = make_tensor(reshape_if_1d(args[1].get_shape()));
        auto y_desc = make_tensor(reshape_if_1d(output_shape));
        
        float alpha=1.0f;
        float beta=0.0f;
        auto status = miopenConvolutionForward(ctx.get_stream().get_miopen(), // miopenHandle_t handle
                                                    &alpha, // const void *alpha
                                                    x_desc.get(), // const miopenTensorDescriptor_t xDesc
                                                    args[0].implicit(), // const void *x
                                                    w_desc.get(),// const miopenTensorDescriptor_t wDesc
                                                    args[1].implicit(),// const void *w
                                                    cd.get(),// const miopenConvolutionDescriptor_t convDesc
                                                    algo,//  miopenConvFwdAlgorithm_t algo
                                                    &beta,// const void *beta
                                                    y_desc.get(),// const miopenTensorDescriptor_t yDesc
                                                    args[3].implicit(),//  void *y
                                                    workspace_arg.implicit(),// void *workSpace
                                                    workspace_arg.get_shape().bytes()); // size_t workSpaceSize

        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("MIOpen Convolution: running convolution failed");
    }
    return args[3];
}

shape miopen_convolution_dynamic::find(context& ctx, const shape& output_shape, std::vector<shape> inputs)
{
    shape workspace_shape{};

    auto x_desc = make_tensor(reshape_if_1d(inputs[0]));
    auto w_desc = make_tensor(reshape_if_1d(inputs[1]));
    auto y_desc = make_tensor(reshape_if_1d(output_shape));

    std::size_t workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(ctx.get_stream().get_miopen(),
                                             w_desc.get(),
                                             x_desc.get(),
                                             cd.get(),
                                             y_desc.get(),
                                             &workspace_size);
    workspace_shape = shape{shape::int8_type, {workspace_size}};

    auto x         = allocate_gpu(inputs[0]);
    auto w         = allocate_gpu(inputs[1]);
    auto y         = allocate_gpu(output_shape);
    auto workspace = allocate_gpu(workspace_shape);

    int algo_count = 1;
    miopenConvAlgoPerf_t perf;
    auto status = miopenFindConvolutionForwardAlgorithm(ctx.get_stream().get_miopen(),
                                                        x_desc.get(),
                                                        x.implicit(),
                                                        w_desc.get(),
                                                        w.implicit(),
                                                        cd.get(),
                                                        y_desc.get(),
                                                        y.implicit(),
                                                        1,
                                                        &algo_count,
                                                        &perf,
                                                        workspace.implicit(),
                                                        workspace_size,
                                                        false);
    if(status != miopenStatusSuccess)
        MIGRAPHX_THROW("MIOpen Convolution: find convolution failed");
    algo = perf.fwd_algo;

    // add buffer
    std::string input_key;
    for(int i=0;i<inputs[0].lens().size();++i)
    {
        input_key+=std::to_string(inputs[0].lens()[i]);
    }
    if(perf_buffer.find(input_key)==perf_buffer.end())
    {
        perf_buffer[input_key]=perf;
    }

    return shape{shape::int8_type, {perf.memory}};
}

void miopen_convolution_dynamic::finalize(context& ctx,
                                  const shape& output_shape,
                                  std::vector<shape> inputs)
{
    shape kernel_shape=inputs[1];

    bool is_conv_1d=(kernel_shape.lens().size()==3&&op.padding.size()==2&&op.padding[0]==op.padding[1]&&op.padding[0]==0); // 如果是1维卷积且pad是对称的
    bool is_conv_2d=(kernel_shape.lens().size()==4&&op.padding.size()==4&&op.padding[0]==op.padding[2]&&op.padding[1]==op.padding[3]&&op.padding[0]==0&&op.padding[1]==0);
    
    if(kernel_shape.type()==shape::float_type&&(is_conv_1d||is_conv_2d))
    {
        return;
    }
    else
    {
        // 首次需要分配workspace
        if(is_first_finalize)
        {
            cd = make_conv(op);
            workspace_arg=allocate_gpu(inputs.at(2),false);
            is_first_finalize=false;
        }
        else
        {
            // 如果不是首次，则需要先查找缓存，如果缓存没有，则需要计算workspace
            shape ws;
            std::string input_key;
            for(int i=0;i<inputs[0].lens().size();++i)
            {
                input_key+=std::to_string(inputs[0].lens()[i]);
            }
            if(perf_buffer.find(input_key)!=perf_buffer.end())
            {
                miopenConvAlgoPerf_t perf=perf_buffer[input_key];
                algo=perf.fwd_algo;
                ws=shape{shape::int8_type, {perf.memory}};
            }
            else
            {
                ws=find(ctx, output_shape, inputs);
            }
            
            if(ws.bytes() > workspace_arg.get_shape().bytes())
            {
                workspace_arg=allocate_gpu(ws,false);
            }

        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
