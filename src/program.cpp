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

#include <migraphx/program.hpp>
#include <migraphx/version.h>
#include <migraphx/onnx.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/iterator.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/op/load.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/marker.hpp>
#include <migraphx/supported_segments.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/env.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <set>
#include <utility>

#include <unordered_set>
#include <map>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using milliseconds = std::chrono::duration<double, std::milli>;

struct program_impl
{
    // A map is used to keep references to modules of the program
    std::unordered_map<std::string, module> modules;
    context ctx;
    std::string target_name;
    std::size_t device_id;
    bool offload_copy = false;
};

program::program() : impl(std::make_unique<program_impl>()) { this->create_module("main"); }

program::program(program&&) noexcept = default;
program::~program() noexcept         = default;

// copy constructor
program::program(const program& p) { assign(p); }

// copy assignment operator
program& program::operator=(program p)
{
    std::swap(p.impl, this->impl);
    return *this;
}

void program::assign(const program& p)
{
    if(not impl)
    {
        impl = std::make_unique<program_impl>();
    }
    else if(not impl->modules.empty())
    {
        impl->modules.clear();
    }

    impl->ctx         = p.impl->ctx;
    impl->target_name = p.impl->target_name;
    impl->modules     = p.impl->modules;

    // build a map from old ins to new ins
    // Build a map from old module to new module
    std::unordered_map<module_ref, module_ref> mod_map;
    std::transform(
        impl->modules.begin(),
        impl->modules.end(),
        std::inserter(mod_map, mod_map.begin()),
        [&](auto&& xp) { return std::make_pair(&p.impl->modules.at(xp.first), &xp.second); });

    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(auto&& pp : mod_map)
    {
        auto old_ins = iterator_for(*pp.first);
        auto new_ins = iterator_for(*pp.second);
        std::transform(old_ins.begin(),
                       old_ins.end(),
                       new_ins.begin(),
                       std::inserter(ins_map, ins_map.begin()),
                       [](auto x, auto y) { return std::make_pair(x, y); });
    }

    // Update all references from all modules
    for(auto&& mp : impl->modules)
    {
        for(auto ins : iterator_for(mp.second))
            instruction::replace_refs(ins, ins_map, mod_map);
    }
}

shape program::get_parameter_shape(std::string name)
{
    auto* mm = this->get_main_module();
    return mm->get_parameter_shape(std::move(name));
}

std::vector<std::string> program::get_parameter_names()
{
    auto* mm = this->get_main_module();
    return mm->get_parameter_names();
}

instruction_ref program::get_parameter(std::string name)
{
    auto* mm = this->get_main_module();
    return mm->get_parameter(std::move(name));
}

std::unordered_map<std::string, shape> program::get_parameter_shapes()
{
    auto* mm = this->get_main_module();
    return mm->get_parameter_shapes();
}

std::size_t program::size() const { return impl->modules.size(); }

std::vector<shape> program::get_output_shapes() const
{
    const auto* mm = this->get_main_module();
    return mm->get_output_shapes();
}

std::unordered_map<std::string, shape> program::get_inputs() const
{
    const auto* mm = this->get_main_module();
    return mm->get_input_shapes();
}

std::unordered_map<std::string, shape> program::get_outputs() const
{
    const auto* mm = this->get_main_module();
    return mm->output_shapes();
}

std::size_t program::get_memory_usage() const
{
    auto* m                  = this->get_main_module();
    std::size_t memory_usage = 0;
    for(auto ins : iterator_for(*m))
    {
        std::string ins_name = ins->name();
        if(ins_name == "hip::hip_copy_literal" || ins_name == "hip::hip_allocate_memory")
        {
            memory_usage += (ins->get_shape().bytes());
        }
    }
    return memory_usage;
}

int program::get_mxr_version() const
{

    /*
    mxr file version is for the data structure or format of the MXR file. Version should be bumped
    if any changes occur to the format of the MXR file.
    */
    const int mxr_version = 7;

    return mxr_version;
}

context& program::get_context() const { return impl->ctx; }

instruction_ref program::validate() const
{
    const auto* mm = this->get_main_module();
    return mm->validate();
}

target_assignments program::get_target_assignments(const std::vector<target>& targets,
                                                   assignment_options options)
{
    const auto m = options.metric;

    target_assignments p;

    const auto* mod = get_main_module();
    std::vector<std::pair<target, supported_segments>> target_subgraphs;
    target_subgraphs.reserve(targets.size());
    std::transform(targets.begin(),
                   targets.end(),
                   std::back_inserter(target_subgraphs),
                   [&](const auto& t) { return std::make_pair(t, t.find_supported(mod, m)); });

    for(const auto ins : iterator_for(*mod))
    {
        if(contains(p, ins))
        {
            continue;
        }

        for(const auto& [target, subgraph] : target_subgraphs)
        {
            // can't pass a structured binding into lambda in C++17 so create a variable for it
            const auto& t = target;
            for(const auto& segment : subgraph)
            {
                const auto& instructions = segment.instructions;
                if(not contains(instructions, ins))
                {
                    continue;
                }
                std::transform(instructions.begin(),
                               instructions.end(),
                               std::inserter(p, p.end()),
                               [&](auto instr) { return std::make_pair(instr, t.name()); });
            }
        }
    }
    return p;
}

bool program::is_compiled() const { return not this->impl->target_name.empty(); }

void program::compile(const target& t, compile_options options)
{
    assert(not this->is_compiled());
    this->impl->device_id    = options.device_id;
    this->impl->target_name  = t.name();
    this->impl->ctx          = t.get_context(this->impl->device_id);
    this->impl->offload_copy = options.offload_copy;
    options.is_dynamic       = this->get_main_module()->get_dynamic();
    if(enabled(MIGRAPHX_TRACE_COMPILE{}))
        options.trace = tracer{std::cout};

    options.trace(*this);
    options.trace();

    auto&& passes = t.get_passes(this->impl->ctx, options);
    run_passes(*this, passes, options.trace);

    auto mods = this->get_modules();

    // Validate and finalize
    for(const auto& mod : reverse(mods))
    {
        auto invalid = mod->validate();
        if(invalid != mod->end())
        {
            MIGRAPHX_THROW("Invalid module " + mod->name() + " from compilation at instruction " +
                           std::to_string(std::distance(mod->begin(), invalid)));
        }
        auto dangling = mod->find_dangling_reference();
        if(dangling != mod->end())
        {
            auto index = std::distance(mod->begin(), dangling);
            MIGRAPHX_THROW("Dangling reference in module " + mod->name() + " from instruction " +
                           std::to_string(index));
        }
        mod->finalize(this->impl->ctx);
    }
}

void program::finalize()
{
    auto* mm = this->get_main_module();
    mm->finalize(this->impl->ctx);
}

template <class T>
std::string classify(T x)
{
    switch(std::fpclassify(x))
    {
    case FP_INFINITE: return "inf";
    case FP_NAN: return "nan";
    case FP_NORMAL: return "normal";
    case FP_SUBNORMAL: return "subnormal";
    case FP_ZERO: return "zero";
    default: return "unknown";
    }
}

std::unordered_set<std::string> classify_argument(const argument& a)
{
    std::unordered_set<std::string> result;
    a.visit(
        [&](auto t) {
            for(const auto& x : t)
                result.insert(classify(x));
        },
        [&](const auto& xs) {
            for(const auto& x : xs)
            {
                auto r = classify_argument(x);
                result.insert(r.begin(), r.end());
            }
        });
    return result;
}

void preview_argument(std::ostream& os, const argument& a)
{
    a.visit(
        [&](auto t) {
            if(t.size() <= 10)
            {
                os << t;
            }
            else
            {
                os << to_string_range(t.begin(), t.begin() + 5);
                os << ", ..., ";
                os << to_string_range(t.end() - 5, t.end());
                os << "(size:" << t.end() - t.begin() << ")";
            }
        },
        [&](const auto& xs) {
            for(const auto& x : xs)
            {
                os << '{';
                preview_argument(os, x);
                os << '}';
            }
        });
}

static void Reshape(const module* mod,
                    instruction_ref ins,
                    const std::unordered_map<std::string, std::vector<std::size_t>>& inputs,
                    std::unordered_map<instruction_ref, argument>& results,
                    context& ctx)
{
    std::string name = ins->name();

    if(name == "@param")
    {
        // 修改网络的输入大小
        bool changed           = false;
        std::string param_name = any_cast<builtin::param>(ins->get_operator()).parameter;

        // 处理offload==false的情况
        if(contains(param_name, "#output_"))
        {
            return;
        }

        std::unordered_map<std::string, std::vector<std::size_t>>::const_iterator iter =
            inputs.find(param_name);

        // 找到了网络输入
        if(iter != inputs.end())
        {
            // 判断输入大小是否有改变
            std::vector<std::size_t> old_input_shape = ins->get_shape().lens();
            std::vector<std::size_t> new_input_shape = iter->second;
            for(int i = 0; i < old_input_shape.size(); ++i)
            {
                if(old_input_shape[i] != new_input_shape[i])
                {
                    changed = true;
                    break;
                }
            }

            // 如果输入大小确实改变
            if(changed)
            {
                // 获取最大输入shape
                shape max_shape = mod->get_input_shape(param_name);

                int new_elements = std::accumulate(new_input_shape.begin(),
                                                   new_input_shape.end(),
                                                   std::size_t{1},
                                                   std::multiplies<std::size_t>());

                // 判断是否超出了最大输入大小
                if(new_elements > max_shape.elements())
                {
                    MIGRAPHX_THROW("input shape {" + to_string_range(new_input_shape) +
                                   "} exceed the max input shape {" +
                                   to_string_range(max_shape.lens()) + "}");
                }
                else
                {
                    // 如果大小没有超，则修改网络输入大小
                    ins->set_output_shape(shape(ins->get_shape().type(), new_input_shape));
                }
            }
            else
            {
                // 如果输入大小没变，则不需要reshape
                return;
            }
        }
    }

    // 对每条指令执行reshape
    bool need_reshape = ins->reshape(ins, results);

    // 修改results中保存的输出tensor的shape
    if(need_reshape && ins->name() != "hip::sync_stream" && ins->inputs().size() > 1)
    {
        // 每条指令对应的输出tensor
        instruction_ref output_instruction = ins->inputs()[ins->inputs().size() - 1];

        results[output_instruction] =
            results[output_instruction].reshape(output_instruction->get_shape());
    }

    // 最后需要执行finalize
    ins->finalize(ctx);
}

// #define DYNAMIC_SHAPE_DEBUG

template <class F>
std::vector<argument> generic_eval_dynamic(const module* mod,
                                           context& ctx,
                                           std::unordered_map<std::string, argument> params,
                                           std::unordered_map<instruction_ref, argument> results,
                                           F make_trace)
{
    assert(mod->validate() == mod->end());
    results.reserve(mod->size() * 2);
    std::vector<argument> values;
    values.reserve(16);
    auto trace = make_trace(mod);

    // 处理offload_copy为false的params
    std::vector<std::string> model_output_names = mod->get_output_names();
    std::unordered_map<std::string, argument> new_params;
    // 先处理output_name
    for(auto iter = params.begin(); iter != params.end(); ++iter)
    {
        std::string name = iter->first;
        for(int i = 0; i < model_output_names.size(); ++i)
        {
            if(name == model_output_names[i])
            {
                std::string new_name = "main:#output_" + std::to_string(i);
                new_params[new_name] = iter->second;
                break;
            }
        }
    }
    if(new_params.empty())
    {
        new_params = params;
    }
    else
    {
        // 保存输入
        for(auto i : mod->get_input_shapes())
        {
            new_params[i.first] = params[i.first];
        }
    }

    // 获取输入大小
    std::unordered_map<std::string, std::vector<std::size_t>> inputs = {};
    for(auto input_map : params)
    {
        inputs[input_map.first] = input_map.second.get_shape().lens();
    }
    for(auto ins : iterator_for(*mod))
    {
        assert(results.find(ins) == results.end());
        const auto& name = ins->name();

#ifdef DYNAMIC_SHAPE_DEBUG
        // 打印动态模式出错指令
        static int index = 0;
        printf("=======%d: %s=========\n", index++, name.c_str());
#endif
        // 执行指令前进行reshape
        // 这里处理数据无依赖型算子，因为这些算子的输出shape只与输入tensor的shape有关，比如卷积算子
        Reshape(mod, ins, inputs, results, ctx);

        if(name == "@literal")
        {
            results.emplace(ins, trace(ins, [&] { return ins->get_literal().get_argument(); }));
        }
        else if(name == "@param")
        {
            results.emplace(ins, trace(ins, [&] {
                                auto param_name =
                                    any_cast<builtin::param>(ins->get_operator()).parameter;
                                if(not contains(new_params, param_name))
                                    MIGRAPHX_THROW("Parameter not found: " + param_name);
                                auto param = new_params[param_name];
                                return param;
                            }));
        }
        else if(name == "@outline")
        {
            results.emplace(ins, trace(ins, [&] { return argument{ins->get_shape(), nullptr}; }));
        }
        else if(name == "@return")
        {
            std::vector<argument> prog_outputs;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(prog_outputs),
                           [&](instruction_ref i) {
                               assert(results.find(i) != results.end());
                               return results[i];
                           });

            return prog_outputs;
        }
        else
        {
            values.resize(ins->inputs().size());
            std::transform(
                ins->inputs().begin(), ins->inputs().end(), values.begin(), [&](instruction_ref i) {
                    assert(results.find(i) != results.end());
                    return results[i];
                });

            const auto& mod_args = ins->module_inputs();
            auto module_eval     = [&](module_ref smod,
                                   const std::unordered_map<std::string, argument>& inputs) {
                auto ssctx = ctx;
                return generic_eval_dynamic(smod, ssctx, inputs, results, make_trace);
            };

            results.emplace(ins, trace(ins, [&] {
                                return ins->normalized_operator().compute(
                                    ctx, ins->get_shape(), values, mod_args, module_eval);
                            }));
        }

        // 对于数据依赖型算子，由于是在算子的compute()函数中计算输出shape,所以这里需要设置指令的输出大小
        if(ins->name() == "gpu::reshape_dynamic" || ins->name() == "gpu::slice_dynamic" ||
           ins->name() == "gpu::constantofshape" || ins->name() == "gpu::multibroadcast_dynamic" ||
           ins->name() == "gpu::pad_dynamic" || ins->name() == "gpu::range" ||
           ins->name() == "gpu::resize" || ins->name() == "gpu::tile")
        {
            ins->set_output_shape(results[ins].get_shape());
        }

#ifdef DYNAMIC_SHAPE_DEBUG
        // 动态shape调试
        ins->debug_print();
#endif

        assert(results.find(ins) != results.end());
        if(not ins->get_shape().dynamic())
        {
            assert(results.at(ins).get_shape() == ins->get_shape());
        }
    }
    return {results.at(std::prev(mod->end()))};
}

template <class F>
std::vector<argument> generic_eval(const module* mod,
                                   context& ctx,
                                   std::unordered_map<std::string, argument> params,
                                   std::unordered_map<instruction_ref, argument> results,
                                   F make_trace)
{
    assert(mod->validate() == mod->end());
    results.reserve(mod->size() * 2);
    std::vector<argument> values;
    values.reserve(16);
    auto trace = make_trace(mod);

    // 处理offload_copy为false的params
    std::vector<std::string> model_output_names = mod->get_output_names();
    std::unordered_map<std::string, argument> new_params;
    // 先处理output_name
    for(auto iter = params.begin(); iter != params.end(); ++iter)
    {
        std::string name = iter->first;
        for(int i = 0; i < model_output_names.size(); ++i)
        {
            if(name == model_output_names[i])
            {
                std::string new_name = "main:#output_" + std::to_string(i);
                new_params[new_name] = iter->second;
                break;
            }
        }
    }
    if(new_params.empty())
    {
        new_params = params;
    }
    else
    {
        // 保存输入
        for(auto i : mod->get_input_shapes())
        {
            new_params[i.first] = params[i.first];
        }
    }

    for(auto ins : iterator_for(*mod))
    {
        assert(results.find(ins) == results.end());
        const auto& name = ins->name();

        if(name == "@literal")
        {
            results.emplace(ins, trace(ins, [&] { return ins->get_literal().get_argument(); }));
        }
        else if(name == "@param")
        {
            results.emplace(
                ins, trace(ins, [&] {
                    auto param_name = any_cast<builtin::param>(ins->get_operator()).parameter;
                    if(not contains(new_params, param_name))
                        MIGRAPHX_THROW("Parameter not found: " + param_name);
                    auto param = new_params[param_name];
                    
                    /* offload_copy为false的时候，下面的计算图中，如果用户直接使用get_outputs()的方式获取输出，则获取的第一个输出为{1, 96, 96, 2}
                    则为第一个输出节点(对应main:#output_0)分配的tensor大小为{1, 96, 96, 2}，但是实际计算图中main:#output_0的指令大小为{9216, 2}
                    
                    main:#output_0 = @param:main:#output_0 -> float_type, {9216, 2}, {2, 1}
                    main:@580 = gpu::code_object[code_object=13616,symbol_name=softmax_kernel,global=589824,local=64,](main:@579,main:#output_0) -> float_type, {9216, 2}, {2, 1}
                    main:@586 = reshape[dims={1, 96, 96, 2}](main:@580) -> float_type, {1, 96, 96, 2}, {18432, 192, 2, 1}
                    main:@589 = hip::sync_stream(main:@586,main:@588,main:@587) -> float_type, {1, 96, 96, 2}, {18432, 192, 2, 1}
                    main:@590 = @return(main:@589,main:@588,main:@587)
                    */
                    if(!contains(param_name, "#output_"))
                    {
                        if(param.get_shape() != ins->get_shape())
                        {
                            MIGRAPHX_THROW("Incorrect shape {" + to_string(param.get_shape()) +
                                           "} for parameter: " + param_name);
                        }
                    }
                    else
                    {
                        param = param.reshape(ins->get_shape());
                    }
                    return param;
                }));
        }
        else if(name == "@outline")
        {
            results.emplace(ins, trace(ins, [&] { return argument{ins->get_shape(), nullptr}; }));
        }
        else if(name == "@return")
        {
            std::vector<argument> prog_outputs;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(prog_outputs),
                           [&](instruction_ref i) {
                               assert(results.find(i) != results.end());
                               return results[i];
                           });

            return prog_outputs;
        }
        else
        {
            values.resize(ins->inputs().size());
            std::transform(
                ins->inputs().begin(), ins->inputs().end(), values.begin(), [&](instruction_ref i) {
                    assert(results.find(i) != results.end());
                    return results[i];
                });

            const auto& mod_args = ins->module_inputs();
            auto module_eval     = [&](module_ref smod,
                                   const std::unordered_map<std::string, argument>& inputs) {
                auto ssctx = ctx;
                return generic_eval(smod, ssctx, inputs, results, make_trace);
            };

            results.emplace(ins, trace(ins, [&] {
                                return ins->normalized_operator().compute(
                                    ctx, ins->get_shape(), values, mod_args, module_eval);
                            }));
        }
        assert(results.find(ins) != results.end());
        if(not ins->get_shape().dynamic())
        {
            assert(results.at(ins).get_shape() == ins->get_shape());
        }
    }
    return {results.at(std::prev(mod->end()))};
}

template <class F>
std::vector<argument> generic_eval(const program& p,
                                   context& ctx,
                                   std::unordered_map<std::string, argument> params,
                                   F make_trace)
{
    module* mm = const_cast<module*>(p.get_main_module());
    if(mm->get_dynamic())
    {
        return generic_eval_dynamic(mm, ctx, params, {}, make_trace);
    }
    else
    {
        return generic_eval(mm, ctx, params, {}, make_trace);
    }
}

std::vector<argument> program::eval(parameter_map params,
                                    const std::vector<std::string>& output_names) const
{
    auto& ctx = this->impl->ctx;
#ifndef NDEBUG
    auto with_check_context = [&](auto f) {
        return [=, &ctx](auto&&) {
            auto sctx          = std::make_shared<context>(ctx);
            auto check_context = [=, &ctx](auto g) {
                assert(is_shared(ctx, *sctx));
                auto x = g();
                *sctx  = ctx;
                return x;
            };
            return [=](auto&&... xs) { return f(xs..., check_context); };
        };
    };
#else
    auto with_check_context = [](auto f) {
        return [=](auto&&) {
            return [=](auto&&... xs) { return f(xs..., [](auto g) { return g(); }); };
        };
    };
#endif

    auto trace_level = value_of(MIGRAPHX_TRACE_EVAL{});

    std::vector<argument> results;
    if(trace_level > 0)
    {
        std::unordered_map<instruction_ref, std::string> ins_out;
        // get instruction names
        this->print([&](auto x, auto ins_names) {
            std::stringstream ss;
            instruction::print(ss, x, ins_names);
            ins_out[x] = ss.str();
        });

        results = generic_eval(*this,
                               ctx,
                               std::move(params),
                               with_check_context([&](auto& ins, auto f, auto&& check_context) {
                                   ctx.finish();
                                   timer t{};
                                   auto result = check_context(f);
                                   double t1   = t.record<milliseconds>();
                                   ctx.finish();
                                   double t2 = t.record<milliseconds>();

                                   // 重新获取所有reshape后的指令，这样在动态shape中就可以显示指令的reshape后的格式
                                   this->print([&](auto x, auto ins_names) {
                                       std::stringstream ss;
                                       instruction::print(ss, x, ins_names);
                                       ins_out[x] = ss.str();
                                   });
                                   std::cout << "Run instruction: " << ins_out.at(ins) << std::endl;

                                   std::cout << "Time: " << t1 << "ms, " << t2 << "ms" << std::endl;
                                   if(trace_level > 1 and ins->name().front() != '@' and
                                      ins->name() != "load" and not result.empty() and
                                      ins->name() != "gpu::topk")
                                   {
                                       target tgt = make_target(this->impl->target_name);
                                       argument buffer;
                                       if(ins->name() == "gpu::shape_convert") // 直接输出cpu数据
                                       {
                                           buffer = result;
                                       }
                                       else
                                       {
                                           buffer = tgt.copy_from(result);
                                       }
                                       if(trace_level == 2)
                                       {
                                           std::cout << "Output has "
                                                     << to_string_range(classify_argument(buffer))
                                                     << std::endl;
                                           std::cout << "Output: ";
                                           preview_argument(std::cout, buffer);
                                           std::cout << std::endl;
                                       }
                                       else
                                       {
                                           std::cout << "Output: " << buffer << std::endl;
                                       }
                                   }
                                   return result;
                               }));
    }
    else
    {
        results = generic_eval(*this,
                               ctx,
                               std::move(params),
                               with_check_context([&](auto&, auto f, auto&& check_context) {
                                   return check_context(f);
                               }));
    }

    // 按照输出节点输出
    if(output_names.size() == 0)
    {
        return results;
    }
    else
    {
        // 先获取模型输出名
        std::vector<std::string> model_output_names = this->get_main_module()->get_output_names();
        std::vector<int> index;
        for(int i = 0; i < output_names.size(); ++i)
        {
            int j = 0;
            for(j = 0; j < model_output_names.size(); ++j)
            {
                if(model_output_names[j] == output_names[i])
                {
                    index.push_back(j);
                    break;
                }
            }
            if(j == model_output_names.size())
            {
                // 没找到
                printf("%s not found! Program will get all outputs.", output_names[i].c_str());
                return results;
            }
        }

        // 返回指定输出
        std::vector<argument> results2;
        for(int i = 0; i < index.size(); ++i)
        {
            results2.push_back(results[index[i]]);
        }
        return results2;
    }
}

static std::string get_migraphx_version()
{
    std::stringstream ss;
    ss << std::to_string(MIGRAPHX_VERSION_MAJOR) << "." << std::to_string(MIGRAPHX_VERSION_MINOR)
       << "." << std::to_string(MIGRAPHX_VERSION_PATCH);
    return ss.str();
}

value program::to_value() const
{
    value result;
    result["version"]          = get_mxr_version();
    result["migraphx_version"] = get_migraphx_version();
    result["target"]           = this->impl->target_name;
    result["offload_copy"]     = this->impl->offload_copy;
    if(not this->impl->target_name.empty())
        result["context"] = this->impl->ctx.to_value();

    value module_vals = value::object{};
    std::unordered_map<instruction_ref, std::string> names;
    for(auto& mod : this->get_modules())
    {
        value mod_val;
        value nodes;
        mod_val["name"]    = mod->name();
        mod_val["dynamic"] = mod->get_dynamic();

        // 保存输入输出信息，由于每个module都有输入输出，所以每个module都需要保存
        std::vector<std::string> input_names;
        std::vector<value> input_shapes;
        for(auto i : mod->get_input_shapes())
        {
            input_names.push_back(i.first);
            value shape_value;
            migraphx_to_value(shape_value, i.second);
            input_shapes.push_back(shape_value);
        }
        mod_val["input_names"]  = input_names;
        mod_val["input_shapes"] = input_shapes;

        std::vector<std::string> output_names = mod->get_output_names();
        std::vector<value> output_shapes;
        for(auto i : output_names)
        {
            value shape_value;
            migraphx_to_value(shape_value, mod->output_shapes()[i]);
            output_shapes.push_back(shape_value);
        }
        mod_val["output_names"]  = output_names;
        mod_val["output_shapes"] = output_shapes;

        names = mod->print(
            [&](auto ins, auto ins_names) {
                value node;
                node["output"]     = ins_names.at(ins);
                node["name"]       = ins->name();
                node["shape"]      = migraphx::to_value(ins->get_shape());
                node["normalized"] = ins->is_normalized();
                if(ins->name() == "@literal")
                    node["literal"] = migraphx::to_value(ins->get_literal());
                node["operator"] = ins->get_operator().to_value();
                std::vector<std::string> inputs;
                std::transform(ins->inputs().begin(),
                               ins->inputs().end(),
                               std::back_inserter(inputs),
                               [&](auto i) {
                                   assert(contains(ins_names, i));
                                   return ins_names.at(i);
                               });
                node["inputs"]   = inputs;
                auto module_args = ins->module_inputs();
                if(not module_args.empty())
                {
                    std::vector<std::string> module_inputs;
                    std::transform(module_args.begin(),
                                   module_args.end(),
                                   std::back_inserter(module_inputs),
                                   [&](auto mod_ref) { return mod_ref->name(); });
                    node["module_inputs"] = module_inputs;
                }

                nodes.push_back(node);
            },
            names);
        mod_val["nodes"] = nodes;

        module_vals[mod->name()] = mod_val;
    }

    result["modules"] = module_vals;

    return result;
}

static void mod_from_val(module_ref mod,
                         const value& v,
                         std::unordered_map<std::string, instruction_ref>& instructions,
                         const std::unordered_map<std::string, module_ref>& map_mods)
{
    const auto& module_val = v.at(mod->name());
    bool dynamic           = module_val.at("dynamic").to<bool>();
    mod->set_dynamic(dynamic);

    // 设置输入输出信息
    std::vector<std::string> input_names;
    std::vector<shape> input_shapes;
    for(auto i : module_val.at("input_names"))
    {
        input_names.push_back(i.to<std::string>());
    }
    for(auto i : module_val.at("input_shapes"))
    {
        shape s;
        migraphx_from_value(i, s);
        input_shapes.push_back(s);
    }
    for(int i = 0; i < input_names.size(); ++i)
    {
        mod->set_input_shape(input_names[i], input_shapes[i]);
    }

    std::vector<std::string> output_names;
    std::vector<shape> output_shapes;
    for(auto i : module_val.at("output_names"))
    {
        output_names.push_back(i.to<std::string>());
    }
    for(auto i : module_val.at("output_shapes"))
    {
        shape s;
        migraphx_from_value(i, s);
        output_shapes.push_back(s);
    }
    for(int i = 0; i < output_names.size(); ++i)
    {
        mod->set_output_shape(output_names[i], output_shapes[i]);
        mod->set_output_name(output_names[i]);
    }

    for(const value& node : module_val.at("nodes"))
    {
        instruction_ref output;
        auto name       = node.at("name").to<std::string>();
        auto fields     = node.at("operator");
        auto normalized = node.at("normalized").to<bool>();

        if(name == "@param")
        {
            output = mod->insert_parameter(mod->end(),
                                           fields["parameter"].to<std::string>(),
                                           migraphx::from_value<shape>(node.at("shape")));
        }
        else if(name == "@literal")
        {
            output =
                mod->insert_literal(mod->end(), migraphx::from_value<literal>(node.at("literal")));
        }
        else
        {
            auto op = make_op(name, fields);
            std::vector<instruction_ref> inputs;
            std::transform(node.at("inputs").begin(),
                           node.at("inputs").end(),
                           std::back_inserter(inputs),
                           [&](const value& i) {
                               auto i_name = i.to<std::string>();
                               assert(contains(instructions, i_name));
                               return instructions.at(i_name);
                           });

            std::vector<module_ref> module_inputs;
            if(node.contains("module_inputs"))
            {
                std::transform(node.at("module_inputs").begin(),
                               node.at("module_inputs").end(),
                               std::back_inserter(module_inputs),
                               [&](const value& i) { return map_mods.at(i.to<std::string>()); });

                for(auto& smod : module_inputs)
                {
                    mod_from_val(smod, v, instructions, map_mods);
                }
            }

            if(name == "@return")
            {
                output = mod->add_return(inputs);
            }
            else if(module_inputs.empty())
            {
                output = mod->insert_instruction(mod->end(), op, inputs);
            }
            else
            {
                output = mod->insert_instruction(mod->end(), op, inputs, module_inputs);
            }
        }
        output->set_normalized(normalized);
        instructions[node.at("output").to<std::string>()] = output;
    }
}

void program::from_value(const value& v)
{
    auto mxr_version = v.at("version").to<int>();
    if(mxr_version != get_mxr_version())
    {
        MIGRAPHX_THROW("Error: MXR version mismatch. MXR file was created using MXR version: " +
                       std::to_string(mxr_version) +
                       ", while installed MIGraphX is using MXR version: " +
                       std::to_string(get_mxr_version()) +
                       ", Try regenerating MXR file using installed MIGraphX and running again.");
    }

    auto migx_version = v.at("migraphx_version").to<std::string>();
    if(migx_version != get_migraphx_version())
    {
        std::cout << "warning: MXR File was created using MIGraphX version: " << migx_version
                  << ", while installed MIGraphX is at version: " << get_migraphx_version()
                  << ", operators implementation could be mismatched." << std::endl;
    }

    this->impl->target_name = v.at("target").to<std::string>();
    if(not this->impl->target_name.empty())
    {
        target t        = make_target(this->impl->target_name);
        this->impl->ctx = t.get_context(this->impl->device_id);
        this->impl->ctx.from_value(v.at("context"));
    }

    this->impl->offload_copy = v.at("offload_copy").to<bool>();

    auto module_vals = v.at("modules");
    for(const auto& vv : module_vals)
    {
        const auto& name = vv.get_key();
        if(name == "main")
            continue;
        impl->modules.emplace(name, name);
    }
    std::unordered_map<std::string, module_ref> map_mods;
    std::transform(impl->modules.begin(),
                   impl->modules.end(),
                   std::inserter(map_mods, map_mods.end()),
                   [&](auto&& pp) { return std::make_pair(pp.first, &pp.second); });

    std::unordered_map<std::string, instruction_ref> map_insts;
    auto* mm = get_main_module();
    mod_from_val(mm, module_vals, map_insts, map_mods);

    this->finalize();
}

double common_average(const std::vector<double>& v)
{
    std::size_t n = v.size() / 4;
    double total  = std::accumulate(v.begin() + n, v.end() - n, 0.0);
    return total / std::distance(v.begin() + n, v.end() - n);
}

std::string perf_group(const operation& op)
{
    auto attr = op.attributes();
    if(attr.contains("group"))
        return attr.at("group").to<std::string>();
    return op.name();
}

void program::mark(const parameter_map& params, marker&& m)
{
    auto& ctx = this->impl->ctx;
    // Run once by itself
    eval(params);
    ctx.finish();
    // Start marking
    m.mark_start(*this);
    generic_eval(*this, ctx, params, always([&](auto ins, auto f) {
        argument result;
        m.mark_start(ins);
        result = f();
        m.mark_stop(ins);
        return result;
    }));
    m.mark_stop(*this);
}

void program::perf_report(std::ostream& os,
                          std::size_t n,
                          parameter_map params,
                          std::size_t batch) const
{
    auto& ctx = this->impl->ctx;
    // Run once by itself
    eval(params);
    ctx.finish();
    // Run and time entire program
    std::vector<double> total_vec;
    total_vec.reserve(n);
    for(std::size_t i = 0; i < n; i++)
    {
        total_vec.push_back(time<milliseconds>([&] {
            eval(params);
            ctx.finish();
        }));
    }
    std::sort(total_vec.begin(), total_vec.end());
    std::unordered_map<instruction_ref, std::vector<double>> ins_vec;
    // Fill the map
    generic_eval(*this, ctx, params, always([&](auto ins, auto) {
        ins_vec[ins].reserve(n);
        return argument{ins->get_shape(), nullptr};
    }));

    // Run and time each instruction
    for(std::size_t i = 0; i < n; i++)
    {
        generic_eval(*this, ctx, params, always([&](auto ins, auto f) {
            argument result;
            ins_vec[ins].push_back(time<milliseconds>([&] {
                result = f();
                ctx.finish();
            }));
            return result;
        }));
    }
    for(auto&& p : ins_vec)
        std::sort(p.second.begin(), p.second.end());
    // Run and time implicit overhead
    std::vector<double> overhead_vec;
    overhead_vec.reserve(n);
    for(std::size_t i = 0; i < n; i++)
    {
        overhead_vec.push_back(time<milliseconds>([&] { dry_run(params); }));
    }

    double total_time             = common_average(total_vec);
    double rate                   = 1000.0 / total_time;
    double overhead_time          = common_average(overhead_vec);
    double overhead_percent       = overhead_time * 100.0 / total_time;
    double total_instruction_time = 0.0;
    std::unordered_map<std::string, double> op_times;
    std::unordered_map<std::string, std::size_t> op_n;
    for(auto&& p : ins_vec)
    {
        double avg = common_average(p.second);
        op_times[perf_group(p.first->get_operator())] += avg;
        total_instruction_time += avg;
        op_n[perf_group(p.first->get_operator())]++;
    }
    double calculate_overhead_time    = total_time - total_instruction_time;
    double calculate_overhead_percent = calculate_overhead_time * 100.0 / total_time;

    std::unordered_map<instruction_ref, std::string> names;
    this->print(names, [&](auto ins, auto ins_names) {
        instruction::print(std::cout, ins, ins_names);

        // skip return instruction
        if(ins->name() == "@return")
            return;

        double avg     = common_average(ins_vec[ins]);
        double percent = 100.0 * avg / total_instruction_time;
        os << ": " << avg << "ms, " << percent << "%";
        os << std::endl;
    });

    os << std::endl;
    os << "Summary:" << std::endl;
    std::vector<std::tuple<double, std::size_t, std::string>> op_times_sorted;
    std::transform(
        op_times.begin(), op_times.end(), std::back_inserter(op_times_sorted), [&](auto p) {
            auto&& name = p.first;
            return std::make_tuple(p.second, op_n.at(name), name);
        });
    std::sort(op_times_sorted.begin(), op_times_sorted.end(), std::greater<>{});
    for(auto&& [avg, nn, name] : op_times_sorted)
    {
        double percent = 100.0 * avg / total_instruction_time;
        double per_ins = avg / nn;
        os << name << ": " << avg << "ms / " << nn << " = " << per_ins << "ms, " << percent << "%"
           << std::endl;
    }

    os << std::endl;

    batch = get_main_module()->get_input_shapes().begin()->second.lens()[0];
    os << "Batch size: " << batch << std::endl;
    os << "Rate: " << rate * batch << "/sec" << std::endl;
    os << "Total time: " << total_time << "ms" << std::endl;
    os << "Total instructions time: " << total_instruction_time << "ms" << std::endl;
    os << "Overhead time: " << overhead_time << "ms"
       << ", " << calculate_overhead_time << "ms" << std::endl;
    os << "Overhead: " << std::round(overhead_percent) << "%"
       << ", " << std::round(calculate_overhead_percent) << "%" << std::endl;
}

void program::debug_print() const { std::cout << *this << std::endl; }
void program::debug_print(instruction_ref ins) const
{
    std::unordered_map<instruction_ref, std::string> names;
    if(std::any_of(this->impl->modules.begin(), this->impl->modules.end(), [&](const auto& pp) {
           return is_end(pp.second.end(), ins);
       }))
    {
        std::cout << "End instruction" << std::endl;
        return;
    }
    else if(std::none_of(this->impl->modules.begin(),
                         this->impl->modules.end(),
                         [&](const auto& pp) { return pp.second.has_instruction(ins); }))
    {
        std::cout << "Instruction not part of program" << std::endl;
        return;
    }

    std::stringstream ss;
    this->print(names, [&](auto x, auto ins_names) {
        if(x == ins)
        {
            instruction::print(std::cout, x, ins_names);
            std::cout << std::endl;
        }
    });
}

void program::print(
    std::unordered_map<instruction_ref, std::string>& names,
    const std::function<void(instruction_ref, std::unordered_map<instruction_ref, std::string>)>&
        print_func) const
{
    for(const auto& pp : this->impl->modules)
    {
        names = pp.second.print(print_func, names);
    }
}

void program::print(
    const std::function<void(instruction_ref ins,
                             std::unordered_map<instruction_ref, std::string>)>& print_func) const
{
    std::unordered_map<instruction_ref, std::string> names;
    this->print(names, print_func);
}

void program::print_graph(std::ostream& os, bool brief) const
{
    const auto* mm = this->get_main_module();
    mm->print_graph(os, brief);
}

void program::print_cpp(std::ostream& os) const
{
    auto vec_modules = this->get_modules();
    std::unordered_map<instruction_ref, std::string> names;
    os << "migraphx::program p;\n";
    for(auto& mod : vec_modules)
    {
        std::string var_name = "m" + mod->name();
        os << "migraphx::module_ref " << var_name << " = ";
        if(mod->name() == "main")
            os << "p.get_main_module();";
        else
            os << "p.create_module(\"" << mod->name() << "\");";
        os << std::endl;
        names = mod->print_cpp(os, var_name, names);
        os << std::endl;
    }
}

void program::dry_run(std::unordered_map<std::string, argument> params) const
{
    auto& ctx = this->impl->ctx;
    generic_eval(*this, ctx, std::move(params), always([](auto ins, auto&&...) {
        return argument{ins->get_shape(), nullptr};
    }));
}

void program::annotate(std::ostream& os, const std::function<void(instruction_ref)>& a) const
{
    for(auto& pp : this->impl->modules)
    {
        std::cout << pp.first << ":" << std::endl;
        pp.second.annotate(os, a);
    }
}

const module* program::get_module(const std::string& name) const { return &impl->modules.at(name); }

module* program::create_module(const std::string& name)
{
    assert(not contains(impl->modules, name));
    auto r = impl->modules.emplace(name, name);
    return &(r.first->second);
}

module* program::get_module(const std::string& name) { return &impl->modules.at(name); }

module* program::get_main_module() { return get_module("main"); }

const module* program::get_main_module() const { return get_module("main"); }

template <class T>
std::vector<T*> generic_get_modules(T* mm)
{
    std::vector<T*> vec_modules;
    vec_modules.push_back(mm);
    auto sub_modules = mm->get_sub_modules();
    vec_modules.insert(vec_modules.end(), sub_modules.begin(), sub_modules.end());
    return vec_modules;
}

template <class Map, class T, class OutputIterator>
void generic_get_unused_modules(Map& m, const std::vector<T*>& mods, OutputIterator out)
{
    std::unordered_set<std::string> used;
    std::transform(mods.begin(), mods.end(), std::inserter(used, used.end()), [](auto&& mod) {
        return mod->name();
    });
    transform_if(
        m.begin(),
        m.end(),
        out,
        [&](auto&& pp) { return not contains(used, pp.first); },
        [](auto&& pp) { return &pp.second; });
}

std::vector<const module*> program::get_modules() const
{
    auto result = generic_get_modules(this->get_main_module());
    generic_get_unused_modules(impl->modules, result, std::back_inserter(result));
    return result;
}

std::vector<module*> program::get_modules()
{
    auto result = generic_get_modules(this->get_main_module());
    generic_get_unused_modules(impl->modules, result, std::back_inserter(result));
    return result;
}

template <class Module, class Map>
void generic_insert_module_tree(Module* pm, Map& m)
{
    for(auto* sm : pm->get_sub_modules(true))
    {
        m.insert(std::make_pair(sm, pm));
        generic_insert_module_tree(sm, m);
    }
}

std::unordered_multimap<module_ref, module_ref> program::get_module_tree()
{
    std::unordered_multimap<module_ref, module_ref> result;
    generic_insert_module_tree(this->get_main_module(), result);
    return result;
}

template <class Map, class T>
bool is_unused_module(Map& m, const std::vector<T*>& mods, const std::string& name)
{
    bool is_unused = false;
    generic_get_unused_modules(m, mods, make_function_output_iterator([&](auto* mod) {
                                   if(mod->name() == name)
                                       is_unused = true;
                               }));
    return is_unused;
}

template <class Map>
bool references_instruction(Map& m, const instruction& ins, const std::string& name)
{
    return std::any_of(m.begin(), m.end(), [&](auto&& p) {
        if(p.first == name)
            return false;
        return std::any_of(p.second.begin(), p.second.end(), [&](auto&& i) {
            return std::any_of(i.inputs().begin(), i.inputs().end(), [&](auto&& j) {
                return std::addressof(*j) == std::addressof(ins);
            });
        });
    });
}

void program::remove_module(const std::string& name)
{
    // cppcheck-suppress assertWithSideEffect
    assert(is_unused_module(impl->modules, generic_get_modules(this->get_main_module()), name) &&
           "Module used in program");
    assert(std::none_of(
               impl->modules.at(name).begin(),
               impl->modules.at(name).end(),
               [&](auto&& ins) { return references_instruction(impl->modules, ins, name); }) &&
           "Instruction referenced in another module");

    // if an instruction has an input out side of the current module, need to remove
    // the instruction from its input's outputs
    auto& mod = impl->modules.at(name);
    for(auto ins : iterator_for(mod))
    {
        auto inputs = ins->inputs();
        for(auto in : inputs)
        {
            if(not mod.has_instruction(in))
            {
                in->remove_output(ins);
            }
        }
    }

    impl->modules.erase(name);
}

void program::remove_unused_modules()
{
    std::vector<module*> unused;
    generic_get_unused_modules(
        impl->modules, generic_get_modules(this->get_main_module()), std::back_inserter(unused));
    for(auto* m : unused)
        this->remove_module(m->name());
}

program& program::sort()
{
    for(auto& pp : this->impl->modules)
    {
        pp.second.sort();
    }

    return *this;
}

int program::reshape(const std::unordered_map<std::string, std::vector<std::size_t>>& inputs,
                     bool printInfo)
{
    printf("warning: reshape() has been deprecated in MIGraphX3.0.0.\n");

    return 1;
}

void program::set_device_id(std::size_t device_id) { this->impl->device_id = device_id; }

bool program::get_offload_copy() const { return this->impl->offload_copy; }

bool operator==(const program& x, const program& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const program& p)
{
    auto vec_modules = p.get_modules();
    std::unordered_map<instruction_ref, std::string> names;
    for(auto& mod : vec_modules)
    {
        os << "module: \"" << mod->name() << "\"" << std::endl;
        names = mod->print(
            [&](auto ins, auto ins_names) {
                instruction::print(os, ins, ins_names);
                os << std::endl;
            },
            names);
        os << std::endl;
    }

    return os;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
