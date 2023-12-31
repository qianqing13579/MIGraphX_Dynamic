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
#include <migraphx/instruction.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/erase.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/SimpleLog.h>
#include <migraphx/op/load.hpp>
#include <migraphx/op/Shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
auto equal_to(const T& x)
{
    return [&](const T& y) { return std::equal_to<T>{}(x, y); };
}

instruction::instruction(operation o, shape r, std::vector<instruction_ref> args)
    : op(std::move(o)), result(std::move(r)), arguments(std::move(args))
{
}

instruction::instruction(operation o,
                         shape r,
                         std::vector<instruction_ref> args,
                         std::vector<module_ref> modules)
    : op(std::move(o)),
      result(std::move(r)),
      arguments(std::move(args)),
      module_args(std::move(modules))
{
}

instruction::instruction(literal l)
    : op(builtin::literal{}), result(l.get_shape()), lit(std::move(l))
{
}

void instruction::replace(const shape& r)
{
    if(r != result)
    {
        result = r;

        for(auto&& ins : output)
        {
            if(ins->name() == "@return")
                continue;

            assert(ins->name().front() != '@');
            ins->recompute_shape();
        }
    }
}

void instruction::replace(operation o)
{
    normalized = false;
    op         = std::move(o);
    recompute_shape();
}

void instruction::recompute_shape() { replace(compute_shape(op, arguments, module_args)); }

void instruction::clear_arguments()
{
    for(auto&& arg : arguments)
    {
        arg->remove_output(*this);
    }
    arguments.clear();
    module_args.clear();
}

bool operator==(const instruction& i, instruction_ref ref)
{
    return std::addressof(i) == std::addressof(*ref);
}

bool instruction::valid(instruction_ref start, bool check_order) const
{
    return valid() && std::all_of(arguments.begin(), arguments.end(), [&](instruction_ref i) {
               auto self = std::find(i->outputs().begin(), i->outputs().end(), *this);
               bool ret  = self != i->outputs().end();
               if(check_order)
               {
                   // check arguments for this instruction before this instruction
                   ret = ret and (std::distance(start, i) < std::distance(start, *self));
               }
               return ret;
           });
}

bool instruction::valid() const
{
    shape computed;
    if(op.name() == "@literal")
    {
        computed = lit.get_shape();
    }
    else if(op.name() == "@param")
    {
        computed = result;
    }
    else if(op.name() == "@return")
    {
        computed = {};
    }
    else
    {
        try
        {
            computed = compute_shape(op, arguments, module_args);
        }
        catch(migraphx::exception&)
        {
            return false;
        }
    }

    return (result == computed) &&
           std::all_of(output.begin(), output.end(), [&](instruction_ref i) {
               return std::find(i->inputs().begin(), i->inputs().end(), *this) != i->inputs().end();
           });
}

bool instruction::support_dynamic_shape() const
{
    shape result;
    try
    {
        result = compute_shape(op, arguments, module_args);
    }
    catch(migraphx::exception&)
    {
        return false;
    }
    return true;

}

shape instruction::get_shape() const { return result; }
const literal& instruction::get_literal() const
{
    assert(op.name() == "@literal");
    return lit;
}

operation& instruction::get_operator() { return op; }

std::string instruction::name() const { return op.name(); }

const std::vector<instruction_ref>& instruction::inputs() const { return arguments; }

const std::vector<module_ref>& instruction::module_inputs() const { return module_args; }

const std::vector<instruction_ref>& instruction::outputs() const { return output; }

bool operator==(const instruction& x, const instruction& y)
{
    if(not std::equal(x.arguments.begin(),
                      x.arguments.end(),
                      y.arguments.begin(),
                      y.arguments.end(),
                      std::equal_to<instruction_ref>{}))
        return false;
    if(std::tie(x.result, x.op, x.module_args) != std::tie(y.result, y.op, y.module_args))
        return false;
    if(x.name() == "@literal")
        return x.lit == y.lit;
    return true;
}

bool operator!=(const instruction& x, const instruction& y) { return not(x == y); }

bool operator==(instruction_ref ref, const instruction& i) { return i == ref; }

bool operator!=(const instruction& i, instruction_ref ref) { return not(i == ref); }

bool operator!=(instruction_ref ref, const instruction& i) { return not(i == ref); }

void instruction::add_output(instruction_ref ins)
{
    if(std::find_if(output.begin(), output.end(), equal_to(ins)) == output.end())
        output.push_back(ins);
}

void instruction::backreference(instruction_ref ref)
{
    for(auto&& arg : ref->inputs())
        arg->add_output(ref);
}

void instruction::replace_argument(instruction_ref ins,
                                   instruction_ref old,
                                   instruction_ref new_ins)
{
    ins->replace_argument(old, new_ins);
    backreference(ins);
    ins->recompute_shape();
}

void instruction::replace_mod_argument(instruction_ref ins, module_ref old, module_ref new_mod)
{
    ins->replace_mod_argument(old, new_mod);
    backreference(ins);
    ins->recompute_shape();
}

void instruction::replace(instruction_ref ins,
                          operation o,
                          const shape& r,
                          std::vector<instruction_ref> args)
{
    ins->replace(std::move(o), r, std::move(args));
    backreference(ins);
}

void instruction::replace(instruction_ref ins,
                          operation o,
                          const shape& r,
                          std::vector<instruction_ref> args,
                          std::vector<module_ref> module_args)
{
    ins->replace(std::move(o), r, std::move(args), std::move(module_args));
    backreference(ins);
}

void instruction::replace(operation o, const shape& r, std::vector<instruction_ref> args)
{
    normalized = false;
    op         = std::move(o);
    replace(r);
    replace(std::move(args));
}

void instruction::replace(operation o,
                          const shape& r,
                          std::vector<instruction_ref> args,
                          std::vector<module_ref> mdl_args)
{
    op = std::move(o);
    replace(r);
    replace(std::move(args), std::move(mdl_args));
}

void instruction::replace_refs(
    instruction_ref ins,
    const std::unordered_map<instruction_ref, instruction_ref>& map_insts,
    const std::unordered_map<module_ref, module_ref>& map_mods)
{
    const auto& args = ins->inputs();
    for(const auto& arg : args)
    {
        if(contains(map_insts, arg))
        {
            instruction::replace_argument(ins, arg, map_insts.at(arg));
        }
    }

    const auto& module_args = ins->module_inputs();
    if(module_args.empty())
        return;

    for(const auto& mod : module_args)
    {
        if(contains(map_mods, mod))
        {
            instruction::replace_mod_argument(ins, mod, map_mods.at(mod));
        }
    }
}

void instruction::replace(std::vector<instruction_ref> args)
{
    clear_arguments();
    arguments = std::move(args);
}

void instruction::replace(std::vector<instruction_ref> args, std::vector<module_ref> mdl_args)
{
    clear_arguments();
    arguments   = std::move(args);
    module_args = std::move(mdl_args);
}

void instruction::replace_argument(instruction_ref old, instruction_ref new_ins)
{
    assert(std::any_of(arguments.begin(), arguments.end(), equal_to(old)));
    std::replace_if(arguments.begin(), arguments.end(), equal_to(old), new_ins);
    old->remove_output(*this);
}

void instruction::replace_mod_argument(module_ref old, module_ref new_mod)
{
    assert(std::any_of(module_args.begin(), module_args.end(), [&](auto i) { return i == old; }));
    std::replace(module_args.begin(), module_args.end(), old, new_mod);
}

bool instruction::is_undefined() const
{
    if(op.name() == "undefined")
    {
        return true;
    }
    else if(this->inputs().empty())
    {
        return false;
    }
    else
    {
        return std::all_of(this->inputs().begin(), this->inputs().end(), [](auto arg) {
            return arg->is_undefined();
        });
    }
}

bool instruction::can_eval() const
{
    if(op.name() == "@literal")
    {
        return true;
    }
    else if(is_context_free(op))
    {
        return std::all_of(
            this->inputs().begin(), this->inputs().end(), [](auto arg) { return arg->can_eval(); });
    }
    else
    {
        return false;
    }
}

bool instruction::can_eval_for_shape() const
{
    if(op.name() == "@literal" || op.name()=="Shape")
    {
        return true;
    }
    else if(is_context_free(op))
    {
        return std::all_of(
            this->inputs().begin(), this->inputs().end(), [](auto arg) { return arg->can_eval_for_shape(); });
    }
    else
    {
        return false;
    }
} 

argument instruction::eval(bool check_eval)
{
    if(op.name() == "@literal")
    {
        return this->get_literal().get_argument();
    }
    if(is_context_free(op))
    {
        if(check_eval and not this->can_eval())
            return {};
        std::vector<argument> args;
        std::transform(this->inputs().begin(),
                       this->inputs().end(),
                       std::back_inserter(args),
                       [](auto arg) { return arg->eval(false); });
        return normalized_operator().compute(result, args);
    }
    return {};
}

argument instruction::eval_for_shape(bool check_eval)
{
    if(op.name() == "@literal")
    {
        return this->get_literal().get_argument();
    }
    if(op.name() == "Shape")
    {
        argument input_arg=migraphx::generate_argument(this->inputs()[0]->get_shape());
        return any_cast<op::Shape>(this->get_operator()).compute(this->get_shape(),{input_arg});
    }
    if(is_context_free(op))
    {
        if(check_eval and not this->can_eval_for_shape())
            return {};
        std::vector<argument> args;
        std::transform(this->inputs().begin(),
                       this->inputs().end(),
                       std::back_inserter(args),
                       [](auto arg) { return arg->eval_for_shape(false); });
        return normalized_operator().compute(result, args);
    }
    return {};
}

void instruction::finalize(context& ctx)
{
    if(has_finalize(this->op))
        this->op.finalize(ctx, this->get_shape(), to_shapes(this->inputs()));
}

void instruction::print(std::ostream& os,
                        instruction_ref ins,
                        const std::unordered_map<instruction_ref, std::string>& names)
{
    os << names.at(ins) << " = ";

    os << ins->get_operator();

    if(ins->name() == "@literal")
    {
        if(ins->get_literal().get_shape().elements() > 10)
            os << "{ ... }";
        else
            os << "{" << ins->get_literal() << "}";
    }

    if(not ins->inputs().empty())
    {
        char delim = '(';
        for(auto&& arg : ins->inputs())
        {
            std::string arg_name = contains(names, arg) ? names.at(arg) : "?";
            os << delim << arg_name;
            delim = ',';
        }
        os << ")";
    }

    // print module inputs
    if(not ins->module_inputs().empty())
    {
        std::string delim = ", [";
        for(auto&& mod_arg : ins->module_inputs())
        {
            os << delim << mod_arg->name();
            delim = ", ";
        }
        os << "]";
    }

    // skip return instruction shape
    if(ins->name() != "@return")
        os << " -> " << ins->get_shape();
}

static void debug_name(std::ostream& os, instruction& ins)
{
    if(ins.name() == "@literal")
    {
        os << "@literal";
        if(ins.get_literal().get_shape().elements() > 10)
            os << "{ ... }";
        else
            os << "{" << ins.get_literal() << "}";
    }
    else
    {
        os << ins.get_operator();
    }
}

void instruction::debug_print()
{
    debug_name(std::cout, *this);
    std::string delim = "(";
    for(auto arg : this->inputs())
    {
        std::cout << delim;
        debug_name(std::cout, *arg);
        delim = ", ";
    }
    if(not this->inputs().empty())
        std::cout << ")";
    std::cout << " -> " << this->get_shape() << std::endl;
}

instruction_ref instruction::get_output_alias(instruction_ref ins, bool shallow)
{
    auto i = ins->get_operator().output_alias(to_shapes(ins->inputs()));
    if(i < 0)
        return ins;
    if(shallow)
        return ins->inputs().at(i);
    return get_output_alias(ins->inputs().at(i));
}

void instruction::set_normalized(bool value) { normalized = value; }

bool instruction::is_normalized() const { return normalized; }

bool instruction::need_normalization()
{
    return this->get_operator().need_normalization() and not normalized;
}

operation instruction::normalized_operator()
{
    operation o = this->get_operator();
    if(this->need_normalization())
    {
        auto s = this->inputs().front()->get_shape();
        if(not normalize_attributes(o, s.max_lens()))
            return this->get_operator();
    }
    return o;
}

// dynamic shape
void instruction::set_output_shape(const shape &new_shape)
{
    result=new_shape;
}

bool instruction::reshape(instruction_ref ins,std::unordered_map<instruction_ref, argument> &results)
{
    // 修改算子属性,并返回是否需要在cpu上重新计算输出shape
    // 对于builtin算子，或者数据依赖型算子(需要在gpu上计算输出shape),不需要在这里计算输出shape
    bool need_reshape=this->op.do_reshape(ins,results);

    // 需要执行reshape
    if(need_reshape)
    {
        // 重新计算输出shape
        auto new_output_shape=compute_shape(op, arguments, module_args);

        // 修改指令的输出shape
        set_output_shape(new_output_shape);

        // 修改输出tensor指令的shape(如果该指令的输出tensor是通过单独某条指令生成的，则还需要修改输出tensor指令的shape)
        if(ins->name()!="hip::sync_stream")// hip::sync_stream算子的输出tensor不是通过某条指令生成的
        {
            // 修改每条指令的输出tensor指令(输出内存分配指令)
            if(ins->inputs().size()>1)
            {
                // 指令对应的输出tensor指令
                auto output_instruction=ins->inputs()[ins->inputs().size()-1];
                
                if(output_instruction->inputs().size()>0&&output_instruction->name() == "load")
                {
                    // 是load指令,修改load算子属性，并重新计算输出大小
                    any_cast<migraphx::op::load>(output_instruction->get_operator()).s=new_output_shape;
                    output_instruction->set_output_shape(new_output_shape);
                }
                else if(output_instruction->name() == "hip::copy")
                {
                    output_instruction->set_output_shape(new_output_shape);
                }
                else if(output_instruction->name() == "@param")
                {
                    // 是@param指令
                    output_instruction->set_output_shape(new_output_shape);
                }
                else // 输出内存分配指令是"load+其他算子"组成
                {
                    /* SVTR模型中存在如下模式
                        main:@614 = load[offset=126976,end=142336](main:@1) -> float_type, {1, 32, 8, 15}, {3840, 120, 15, 1}
                        main:@615 = transpose[permutation={0, 2, 1, 3}](main:@614) -> float_type, {1, 8, 32, 15}, {3840, 15, 120, 1}
                        main:@616 = gpu::gemm[alpha=1,beta=0,int8_x4_format=1,compute_fp32=0,trans_batch=1](main:@611,main:@613,main:@615) -> float_type, {1, 8, 32, 15}, {3840, 15, 120, 1}
                    */
                    output_instruction->set_output_shape(new_output_shape);
                }
            }
        }
    }

    return need_reshape;
    
    
}

std::vector<shape> to_shapes(const std::vector<instruction_ref>& args)
{
    std::vector<shape> shapes(args.size());
    std::transform(
        args.begin(), args.end(), shapes.begin(), [](instruction_ref i) { return i->get_shape(); });
    return shapes;
}

shape compute_shape(const operation& op, const std::vector<instruction_ref>& args)
{
    return op.compute_shape(to_shapes(args));
}

shape compute_shape(const operation& op,
                    const std::vector<instruction_ref>& args,
                    const std::vector<module_ref>& mods)
{
    if(mods.empty())
    {
        return op.compute_shape(to_shapes(args));
    }
    else
    {
        return op.compute_shape(to_shapes(args), mods);
    }
}

std::vector<shape> try_compute_shape(const operation& op, const std::vector<shape>& inputs)
{
    shape new_shape;
    try
    {
        new_shape = op.compute_shape(inputs);
    }
    catch(...)
    {
        return {};
    }
    return {new_shape};
}

migraphx::instruction* as_address(const instruction_ref& ins) noexcept
{
    return std::addressof(*ins);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
