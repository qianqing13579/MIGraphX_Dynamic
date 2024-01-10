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
#include <iterator>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/op/convolution.hpp>
#include <migraphx/op/deconvolution.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/if_op.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/op/reshape_dynamic.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/op/lstm.hpp>

#include <migraphx/gpu/batch_norm_inference.hpp>
#include <migraphx/gpu/constantofshape.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/convolution.hpp>
#include <migraphx/gpu/convolution_dynamic.hpp>
#include <migraphx/gpu/deconvolution.hpp>
#include <migraphx/gpu/deconvolution_dynamic.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/int8_conv_pack.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/quant_convolution.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/upsample.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/lstm.hpp>
#include <migraphx/gpu/lstm_nomiopen.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/program.hpp>
#include <utility>
#include <functional>
#include <algorithm>
#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct miopen_apply
{
    module* mod          = nullptr;
    const lowering* pass = nullptr;
    std::unordered_map<std::string, std::function<instruction_ref(instruction_ref)>> apply_map{};
    instruction_ref last{};
    bool offload_copy   = false;
    bool int8_x4_format = true;
    bool compute_fp32   = false;

    context& get_context() const
    {
        assert(pass != nullptr);
        assert(pass->ctx != nullptr);
        return *pass->ctx;
    }

    void check_shape(shape x, instruction_ref i)
    {
        assert(x == i->get_shape());
        (void)x;
        (void)i;
    }

    const std::unordered_set<std::string>& get_rocblas_fp32_archs()
    {
        static std::unordered_set<std::string> supported_archs{"gfx908", "gfx90a"};
        return supported_archs;
    }

    void init()
    {
        assert(mod != nullptr);
        assert(pass != nullptr);

#if ROCBLAS_VERSION_MAJOR >= 2 && ROCBLAS_VERSION_MINOR >= 38
        auto& ctx              = get_context();
        const auto device_name = trim(split_string(get_device_name(), ':').front());
        if(contains(get_rocblas_fp32_archs(), device_name))
            compute_fp32 = true;
        rocblas_gemm_flags flag;
        rocblas_query_int8_layout_flag(ctx.get_stream().get_rocblas(), &flag);
        int8_x4_format = (flag == rocblas_gemm_flags_pack_int8x4);
#endif

        offload_copy = (mod->name() == "main") ? pass->offload_copy : false;

        // fuse_pointwise在某些情况存在精度损失，此时可以禁用fuse_pointwise使用动态kernel
        // pointwise算子的静态实现在fuse_pointwise
        // pass中，最终通过src/targets/gpu/jit/pointwise.cpp实现
        add_generic_op("acos");
        add_generic_op("acosh");
        add_generic_op("add");
        add_generic_op("asin");
        add_generic_op("asinh");
        add_generic_op("atan");
        add_generic_op("atanh");
        add_generic_op("ceil");
        add_generic_op("cos");
        add_generic_op("cosh");
        add_generic_op("div");
        add_generic_op("equal");
        add_generic_op("erf");
        add_generic_op("exp");
        add_generic_op("floor");
        add_generic_op("greater");
        add_generic_op("less");
        add_generic_op("log");
        add_generic_op("logical_and");
        add_generic_op("logical_or");
        add_generic_op("logical_xor");
        add_generic_op("max");
        add_generic_op("min");
        add_generic_op("mul");
        add_generic_op("mod");
        add_generic_op("not");
        add_generic_op("pow");
        add_generic_op("prelu");
        add_generic_op("recip");
        add_generic_op("relu");
        add_generic_op("round");
        add_generic_op("rsqrt");
        add_generic_op("sigmoid");
        add_generic_op("sign");
        add_generic_op("sin");
        add_generic_op("sinh");
        add_generic_op("sqdiff");
        add_generic_op("sqrt");
        add_generic_op("sub");
        add_generic_op("tan");
        add_generic_op("tanh");
        add_generic_op("where");
        add_extend_op("abs");
        add_extend_op("clip");
        add_extend_op("convert");
        add_extend_op("elu");
        add_extend_op("leaky_relu");
        add_extend_op("selu");

        add_extend_op("softmax");
        add_extend_op("reduce_max");
        add_extend_op("reduce_mean");
        add_extend_op("reduce_min");
        add_extend_op("reduce_prod");
        add_extend_op("reduce_sum");

        // 动态实现
        if(mod->get_dynamic())
        {
            add_extend_op("Shape");
            add_extend_op("range");
            add_extend_op("constantofshape");

            // 这些算子的静态实现通过src/targets/gpu/jit实现
            add_generic_op("scatternd_none");
            add_extend_op(
                "pad"); // 动态模式中有些模型也可能存在pad算子，比如ModelZoo/Dynamic/tinyyolov2-8_Nx3x416x416.onnx
            add_extend_op("pad_dynamic");
            add_extend_op("concat");
            add_extend_op("gather");

            // veiw算子(产生view,所以不需要分配输出内存)
            add_reshape_dynamic_op();
            add_slice_dynamic_op();
            add_multibroadcast_dynamic_op();

            // 卷积的动态实现
            add_convolution_op_dynamic();
            add_deconvolution_op_dynamic();
            add_quant_convolution_op_dynamic();

            // pooling的动态实现
            add_pooling_op_dynamic();

            add_extend_op("tile");
        }
        // 静态实现
        else
        {
            add_extend_op("pad");

            // 卷积的静态实现
            add_convolution_op();
            add_deconvolution_op();
            add_quant_convolution_op();

            // pooling的静态实现
            add_extend_op("pooling");
        }

        // contiguous算子特殊处理，contiguous虽然属于pointwise算子，但是由于后面的eliminate_contiguous这个pass需要用到contiguous算子
        // 所以这里不能使用jit实现，在fuse_ops这个pass中会使用find_contiguous匹配器将剩下的gpu::contiguous替换为jit实现
        add_generic_op("contiguous");

        // 对于只实现了动态shape的算子，只使用动态实现（可以适用于静态shape和动态shape）
        add_extend_op("argmax");
        add_extend_op("argmin");
        add_extend_op("logsoftmax");
        add_extend_op("lrn");
        add_extend_op("multinomial");
        add_extend_op("nonzero");
        add_extend_op("prefix_scan_sum");
        add_extend_op("reverse");
        add_extend_op("rnn_var_sl_last_output");
        add_extend_op("rnn_var_sl_shift_output");
        add_extend_op("rnn_var_sl_shift_sequence");
        add_extend_op("scatter_none");
        add_extend_op("topk");
        add_extend_op("upsample");
        add_extend_op("resize");
        add_extend_op("scatter_elements");

        add_lstm_op();
        add_batch_norm_inference_op();

        add_gemm_op<op::dot>("dot");
        add_gemm_op<op::quant_dot>("quant_dot");
        add_if_op();
        add_loop_op();
        add_neg_op();
        add_nms_op();
    }

    void copy_params() const
    {
        if(not offload_copy)
            return;

        for(auto ins : iterator_for(*mod))
        {
            if(ins->name() != "@param")
                continue;

            // parameter no outputs, no need to insert copy to gpu
            if(ins->outputs().empty())
                continue;

            auto pos = std::next(ins);
            auto a   = insert_allocation(pos, ins->get_shape());
            auto c   = mod->insert_instruction(pos, make_op("hip::copy_to_gpu"), ins, a);
            mod->replace_instruction(ins, c);
        }

        // return instruction
        auto ret = std::prev(mod->end());
        if(ret->name() == "@return")
        {
            const auto& inputs = ret->inputs();
            int n              = 0;
            // each input of ret need to be copied from gpu to host, and replace
            // output with copy output
            for(const auto& in : inputs)
            {
                std::string id = "hip::copy_from_gpu" + std::to_string(n);
                auto p_output  = mod->insert_instruction(
                    ret,
                    make_op("hip::copy_from_gpu",
                             {{"shape", to_value(in->get_shape())}, {"id", id}}),
                    in);
                ++n;
                instruction::replace_argument(ret, in, p_output);
            }
        }
        // else branch to handle legacy program without the return instruction
        else
        {
            mod->add_instruction(make_op("hip::copy_from_gpu"), ret);
        }
    }

    // lowering的时候先使用apply_map，然后再使用jit实现，所以动态实现需要保存到apply_map中
    void apply()
    {
        init();
        for(auto it = mod->begin(); it != mod->end(); it++)
        {
            auto s = it->get_shape();
            if(apply_map.count(it->name()) > 0)
            {
                check_shape(s, apply_map.at(it->name())(it));
            }
            else if(has_compiler_for(it->name()))
            {
                check_shape(s, insert_precompile_op(it));
            }
        }

        copy_params();
    }

    instruction_ref insert_precompile_op(instruction_ref ins) const
    {
        auto output                       = insert_allocation(ins, ins->get_shape());
        std::vector<instruction_ref> refs = ins->inputs();
        refs.push_back(output);

        return mod->replace_instruction(
            ins,
            make_op("gpu::precompile_op", {{"op", to_value(ins->get_operator())}}),
            refs,
            ins->module_inputs());
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s) const
    {
        return mod->insert_instruction(ins, make_op("allocate", {{"shape", to_value(s)}}));
    }

    void add_convolution_op()
    {
        apply_map.emplace("convolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::convolution>(ins->get_operator());

            auto conv = miopen_convolution{op, make_conv(op)};
            auto ws   = conv.find(get_context(), ins->get_shape(), to_shapes(ins->inputs()));

            auto workspace = insert_allocation(ins, ws);
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(
                ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
        });
    }

    void add_convolution_op_dynamic()
    {
        apply_map.emplace("convolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::convolution>(ins->get_operator());

            auto conv = miopen_convolution_dynamic{op, make_conv(op, true)};
            auto ws   = conv.find(get_context(), ins->get_shape(), to_shapes(ins->inputs()));

            auto workspace = insert_allocation(ins, ws);
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(
                ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
        });
    }

    void add_pooling_op_dynamic()
    {
        apply_map.emplace("pooling", [=](instruction_ref ins) {
            auto&& op                         = ins->get_operator();
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return mod->replace_instruction(
                ins, make_op("gpu::pooling_dynamic", op.to_value()), refs);
        });
    }

    void add_deconvolution_op()
    {
        apply_map.emplace("deconvolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::deconvolution>(ins->get_operator());

            auto conv = miopen_deconvolution{op, make_deconv(op)};
            auto ws   = conv.find(get_context(), ins->get_shape(), to_shapes(ins->inputs()));

            auto workspace = insert_allocation(ins, ws);
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(
                ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
        });
    }

    void add_deconvolution_op_dynamic()
    {
        apply_map.emplace("deconvolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::deconvolution>(ins->get_operator());

            auto conv = miopen_deconvolution_dynamic{op, make_deconv(op, true)};
            auto ws   = conv.find(get_context(), ins->get_shape(), to_shapes(ins->inputs()));

            auto workspace = insert_allocation(ins, ws);
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(
                ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
        });
    }

    template <typename Op>
    void add_gemm_op(const std::string& name)
    {
        apply_map.emplace(name, [=](instruction_ref ins) {
            std::vector<instruction_ref> refs = ins->inputs();
            assert(refs.size() == 2);
            auto output = insert_allocation(ins, ins->get_shape());
            refs.push_back(output);
            return mod->replace_instruction(
                ins, rocblas_gemm<Op>{Op{}, 1, 0, int8_x4_format, compute_fp32}, refs);
        });
    }

    void add_quant_convolution_op()
    {
        apply_map.emplace("quant_convolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::quant_convolution>(ins->get_operator());
            shape ws;
            miopen_quant_convolution conv;
            auto compile_quant_conv_with_format = [&](bool format) {
                conv = miopen_quant_convolution{op, format, make_conv(op)};
                ws   = conv.find(get_context(), ins->get_shape(), to_shapes(ins->inputs()));
            };

            try
            {
                compile_quant_conv_with_format(int8_x4_format);
            }
            catch(migraphx::exception&)
            {
                // In case no solver supports the default format, retry using the other format.
                compile_quant_conv_with_format(not int8_x4_format);
            }

            auto args      = ins->inputs();
            auto workspace = insert_allocation(ins, ws);
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(ins, conv, args[0], args[1], workspace, output);
        });
    }

    void add_quant_convolution_op_dynamic()
    {
        apply_map.emplace("quant_convolution", [=](instruction_ref ins) {
            auto&& op = any_cast<op::quant_convolution>(ins->get_operator());

            op::convolution conv_op{op.padding,
                                    op.stride,
                                    op.dilation,
                                    op.group,
                                    op.padding_mode,
                                    op.use_dynamic_same_auto_pad};
            auto conv = miopen_convolution_dynamic{conv_op, make_conv(conv_op, true)};
            auto ws   = conv.find(get_context(), ins->get_shape(), to_shapes(ins->inputs()));

            auto workspace = insert_allocation(ins, ws);
            auto output    = insert_allocation(ins, ins->get_shape());

            return mod->replace_instruction(
                ins, conv, ins->inputs().at(0), ins->inputs().at(1), workspace, output);
        });
    }

    void add_lstm_op()
    {
        apply_map.emplace("lstm", [=](instruction_ref ins) {
            auto&& op                           = any_cast<op::lstm>(ins->get_operator());
            std::vector<instruction_ref> inputs = ins->inputs();

            bool has_bias = false;
            if(inputs.size() >= 4 && inputs[3]->name() != "undefined")
            {
                has_bias = true;
            }
            //////////////////// gpu miopen lstm ////////////////
            // auto lstm = miopen_lstm{op, make_lstm(op,inputs),has_bias};

            //////////////////// gpu no_miopen lstm ////////////////
            std::vector<argument> args_cpu; // 保存参数(cpu数据)
            for(int i = 0; i < inputs.size(); ++i)
            {
                if(inputs[i]->can_eval())
                {
                    args_cpu.push_back(inputs[i]->eval());
                }
                else if(inputs[i]->name() == "gpu::constantofshape")
                {
                    auto&& constantofshape =
                        any_cast<gpu::hip_constantofshape>(inputs[i]->get_operator());

                    // 生成常量
                    auto type         = constantofshape.op.value.get_shape().type();
                    migraphx::shape s = constantofshape.op.output_shape;
                    migraphx::argument result{s};
                    constantofshape.op.value.visit([&](auto val) {
                        auto data       = val.front();
                        using data_type = decltype(data);
                        // using data_type = std::remove_cv_t<typename decltype(val)::value_type>;
                        for(int i = 0; i < s.elements(); ++i)
                        {
                            ((data_type*)result.data())[i] = data;
                        }
                    });
                    args_cpu.push_back(result);
                }
            }
            auto lstm = lstm_nomiopen{op, has_bias, args_cpu};

            //////////////////// cpu lstm ////////////////
            // std::vector<argument> args_cpu; // 对于cpu版本的lstm,需要拷贝输入参数
            // args_cpu.resize(inputs.size()+1);
            // for(int i=0;i<inputs.size();++i)
            // {
            //     if(inputs[i]->can_eval())
            //     {
            //         args_cpu[i]=inputs[i]->eval();
            //     }
            //     else
            //     {
            //         args_cpu[i]=argument{inputs[i]->get_shape()};
            //     }
            // }
            // args_cpu[inputs.size()]=argument{ins->get_shape()};
            // auto lstm = lstm_nomiopen{op,has_bias,args_cpu};

            auto output = insert_allocation(ins, ins->get_shape());
            inputs.push_back(output);

            return mod->replace_instruction(ins, lstm, inputs);
        });
    }

    void add_reshape_dynamic_op()
    {
        apply_map.emplace("reshape_dynamic", [=](instruction_ref ins) {
            auto op                             = ins->get_operator();
            std::vector<instruction_ref> inputs = ins->inputs();

            return mod->replace_instruction(
                ins, make_op("gpu::reshape_dynamic", op.to_value()), inputs);
        });
    }
    void add_slice_dynamic_op()
    {
        apply_map.emplace("slice_dynamic", [=](instruction_ref ins) {
            auto op                             = ins->get_operator();
            std::vector<instruction_ref> inputs = ins->inputs();

            return mod->replace_instruction(
                ins, make_op("gpu::slice_dynamic", op.to_value()), inputs);
        });
    }

    void add_multibroadcast_dynamic_op()
    {
        apply_map.emplace("multibroadcast_dynamic", [=](instruction_ref ins) {
            auto op                             = ins->get_operator();
            std::vector<instruction_ref> inputs = ins->inputs();

            return mod->replace_instruction(
                ins, make_op("gpu::multibroadcast_dynamic", op.to_value()), inputs);
        });
    }

    // add_generic_op just constructs the operator with no fields whereas add_extend_op copies over
    // the fields Since it doesn't have fields its default constructed

    void add_generic_op(const std::string& name) { add_generic_op(name, "gpu::" + name); }

    void add_generic_op(const std::string& op_name, const std::string& gpu_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return mod->replace_instruction(ins, make_op(gpu_name), refs);
        });
    }

    void add_extend_op(const std::string& name) { add_extend_op(name, "gpu::" + name); }

    void add_extend_op(const std::string& op_name, const std::string& gpu_name)
    {
        apply_map.emplace(op_name, [=](instruction_ref ins) {
            auto&& op                         = ins->get_operator();
            auto output                       = insert_allocation(ins, ins->get_shape());
            std::vector<instruction_ref> refs = ins->inputs();
            refs.push_back(output);

            return mod->replace_instruction(ins, make_op(gpu_name, op.to_value()), refs);
        });
    }

    void add_batch_norm_inference_op()
    {
        apply_map.emplace("batch_norm_inference", [=](instruction_ref ins) {
            auto&& op       = any_cast<op::batch_norm_inference>(ins->get_operator());
            auto output     = insert_allocation(ins, ins->get_shape());
            shape old_shape = ins->inputs().at(1)->get_shape();
            auto input      = ins->inputs()[0];
            auto input_lens = input->get_shape().lens();
            std::vector<int64_t> rsp_lens(input_lens.size(), 1);
            // for per_activation case, also need to reshape input
            if(op.bn_mode == op::batch_norm_inference::per_activation)
            {
                std::copy(input_lens.begin() + 1, input_lens.end(), rsp_lens.begin() + 1);
            }
            else
            {
                rsp_lens[1] = static_cast<int64_t>(old_shape.elements());
            }

            auto reshape_op = op::reshape{rsp_lens};
            std::vector<instruction_ref> reshapes;
            std::transform(ins->inputs().begin() + 1,
                           ins->inputs().end(),
                           std::back_inserter(reshapes),
                           [&](auto i) { return mod->insert_instruction(ins, reshape_op, i); });

            return mod->replace_instruction(ins,
                                            miopen_batch_norm_inference{op},
                                            input,
                                            reshapes[0],
                                            reshapes[1],
                                            reshapes[2],
                                            reshapes[3],
                                            output);
        });
    }

    // use 0 - input to represent neg
    void add_neg_op()
    {
        apply_map.emplace("neg", [=](instruction_ref ins) {
            auto s = ins->get_shape();
            std::vector<float> zeros(s.elements(), 0.0f);
            auto l0     = mod->add_literal(literal(s, zeros));
            auto output = insert_allocation(ins, s);
            return mod->replace_instruction(
                ins, make_op("gpu::sub"), l0, ins->inputs().front(), output);
        });
    }

    // add input and output argument for the if operator
    void add_if_op()
    {
        apply_map.emplace("if", [=](instruction_ref ins) {
            std::vector<instruction_ref> inputs = ins->inputs();
            auto cpu_cond =
                mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), inputs.front());
            auto sync_cond = mod->insert_instruction(ins, make_op("hip::sync_stream"), cpu_cond);
            inputs.front() = sync_cond;

            return mod->replace_instruction(ins, ins->get_operator(), inputs, ins->module_inputs());
        });
    }

    // replace the loop operator with gpu_loop operator
    void add_loop_op()
    {
        apply_map.emplace("loop", [=](instruction_ref ins) {
            std::vector<instruction_ref> inputs = ins->inputs();
            // copy max_iter from gpu to cpu
            auto cpu_max_iter =
                mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), inputs.at(0));
            auto cpu_cond =
                mod->insert_instruction(ins, make_op("hip::copy_from_gpu"), inputs.at(1));
            auto synced_max_iter =
                mod->insert_instruction(ins, make_op("hip::sync_stream"), cpu_max_iter, cpu_cond);
            inputs.at(0)     = synced_max_iter;
            inputs.at(1)     = cpu_cond;
            auto copy_inputs = inputs;
            std::transform(copy_inputs.begin(),
                           copy_inputs.end(),
                           std::back_inserter(inputs),
                           [&](auto in) { return insert_allocation(ins, in->get_shape()); });

            auto mod_args = ins->module_inputs();
            auto output   = insert_allocation(ins, ins->get_shape());

            const auto* sub_mod = mod_args.front();
            auto cond_out       = insert_allocation(ins, sub_mod->get_output_shapes().front());

            // add cond and mod outputs to the argument list
            inputs.push_back(cond_out);
            inputs.push_back(output);

            return mod->replace_instruction(
                ins, make_op("gpu::loop", ins->get_operator().to_value()), inputs, mod_args);
        });
    }

    void add_nms_op()
    {
        apply_map.emplace("nonmaxsuppression", [=](instruction_ref ins) {
            auto s      = ins->get_shape();
            auto output = insert_allocation(ins, s);
            std::vector<instruction_ref> cpu_inputs;
            auto inputs = ins->inputs();
            int n       = 0;
            std::transform(
                inputs.begin(), inputs.end(), std::back_inserter(cpu_inputs), [&](auto in) {
                    std::string id = "hip::copy_from_gpu_nonmaxsuppression" + std::to_string(n);
                    ++n;
                    return mod->insert_instruction(
                        ins,
                        make_op("hip::copy_from_gpu",
                                {{"shape", to_value(in->get_shape())}, {"id", id}}),
                        in);
                });
            cpu_inputs.front() =
                mod->insert_instruction(ins, make_op("hip::sync_stream"), cpu_inputs);
            auto cpu_out = mod->insert_instruction(ins, ins->get_operator(), cpu_inputs);
            auto gpu_out =
                mod->insert_instruction(ins, make_op("hip::copy_to_gpu"), cpu_out, output);
            return mod->replace_instruction(ins, gpu_out);
        });
    }
};

void lowering::apply(module& m) const { miopen_apply{&m, this}.apply(); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
