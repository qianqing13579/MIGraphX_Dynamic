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
#include <migraphx/gpu/deconvolution.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/generate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape miopen_deconvolution::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(4).standard();
    std::vector<shape> conv_inputs(inputs.begin(), inputs.begin() + 2);
    check_shapes{conv_inputs, *this}.max_ndims(5);
    return op.compute_shape(conv_inputs);
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

argument miopen_deconvolution::compute(context& ctx,
                                     const shape& output_shape,
                                     const std::vector<argument>& args) const
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
        MIGRAPHX_THROW("MIOpen Deconvolution: running convolution failed");
    return args[3];
}

shape miopen_deconvolution::find(context& ctx, const shape& output_shape, std::vector<shape> inputs)
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

void miopen_deconvolution::finalize(context& ctx,
                                  const shape& output_shape,
                                  std::vector<shape> inputs)
{
    // 首次需要分配workspace
    if(is_first_finalize)
    {
        if(cd == nullptr)
        {
            cd = make_deconv(op);
        }
        shape ws   = find(ctx, output_shape, inputs); // 加载mxr静态模型的时候，需要执行find
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

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
