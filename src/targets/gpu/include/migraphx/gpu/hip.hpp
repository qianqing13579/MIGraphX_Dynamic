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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_HIP_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_HIP_HPP

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

std::string hip_error(int error);

argument allocate_gpu(const shape& s, bool host = false);

argument register_on_gpu(const argument& arg);

argument to_gpu(const argument& arg, bool host = false);

argument from_gpu(const argument& arg);

void set_device(std::size_t id);

void gpu_sync();
void gpu_sync(const context& ctx);

void gpu_copy(context& ctx, const argument& src, const argument& dst);
void copy_to_gpu(context& ctx, const argument& src, const argument& dst);
void copy_from_gpu(context& ctx, const argument& src, const argument& dst);

argument get_preallocation(context& ctx, const std::string& id);
std::unordered_map<std::string, argument>&
store_preallocated_param(context& ctx, const std::string& id, const argument& a);

struct hip_allocate
{
    shape s;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"));
    }

    std::string name() const { return "hip::allocate"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return s;
    }
    argument compute(context&, const shape& output_shape, const std::vector<argument>&) const
    {
        return allocate_gpu(output_shape);
    }

    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        (void)ins;
        return false;
    }
};

struct hip_sync_stream
{

    std::string name() const { return "hip::sync_stream"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }

    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        gpu_sync(ctx);
        if(args.empty())
            return {};
        return args.front();
    }
    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        (void)ins;
        return true;
    }
};

struct hip_copy_to_gpu
{
    std::string name() const { return "hip::copy_to_gpu"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1, 2);
        return inputs.at(0);
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        auto input = register_on_gpu(args[0]);
        if(args.size() == 1)
            return input;
        argument result = args[1].share();
        gpu_copy(ctx, input, result);
        // Associate the input since it was registered with hip
        return {result.get_shape(), [input, result]() mutable { return result.data(); }};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>& args) const
    {
        if(args.size() == 1)
            return -1;
        return 1;
    }
};

struct hip_copy_from_gpu
{
    shape s; // 需要分配的内存大小
    std::string id{};
    bool is_first = true;

    std::string name() const { return "hip::copy_from_gpu"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1, 2).standard();
        return inputs.at(0);
    }
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.id, "id"));
    }
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        if(args.size() == 1)
        {
            // 内存提前申请
            argument a = get_preallocation(ctx, id);
            gpu_copy(ctx, args[0], a);

            return a;
        }
        copy_from_gpu(ctx, args[0], args[1]);
        return args[1];
    }
    std::ptrdiff_t output_alias(const std::vector<shape>& args) const
    {
        if(args.size() == 1)
            return -1;
        return 1;
    }
    void finalize(context& ctx, const shape&, const std::vector<shape>&)
    {
        if(is_first == true)
        {
            argument a = allocate_gpu(s, true);
            store_preallocated_param(ctx, id, a);
            is_first = false;
        }
        else
        {
            argument a = get_preallocation(ctx, id);
            store_preallocated_param(ctx, id, a.reshape(s));
        }
    }

    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        // 修改算子属性，在这里修改是为了避免在运行期对输出tensor进行reshape
        s = ins->inputs()[0]->get_shape();
        return true;
    }
};

struct hip_copy
{
    std::string name() const { return "hip::copy"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs.at(0);
    }
    argument compute(context& ctx, const shape&, std::vector<argument> args) const
    {
        gpu_copy(ctx, args[0], args[1]);
        return args[1];
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 1; }
};

struct hip_allocate_memory
{
    shape s;
    std::string id{};
    bool is_first = true;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.id, "id"));
    }

    std::string name() const { return "hip::hip_allocate_memory"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return s;
    }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        return get_preallocation(ctx, id);
    }

    void finalize(context& ctx, const shape&, const std::vector<shape>&)
    {
        if(is_first == true)
        {
            argument a = allocate_gpu(s);
            store_preallocated_param(ctx, id, a);
            is_first = false;
        }
    }
    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        (void)ins;
        return false;
    }
};

struct hip_copy_literal
{
    literal l;
    std::string id{};
    std::unordered_map<std::string, argument> arg_map;

    bool is_first = true; // reshape()的时候，调用finalize不用再进行分配了

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.l, "literal"), f(self.id, "id"));
    }

    std::string name() const { return "hip::hip_copy_literal"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return l.get_shape();
    }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        std::unordered_map<std::string, argument>::const_iterator iter = arg_map.find(id);
        return iter->second;
    }

    // reshape arg_map 中保存的常量参数,可以避免动态shape中发生数据拷贝
    void reshape_param(const shape& new_shape)
    {
        argument new_arg = arg_map[id].reshape(new_shape);
        arg_map[id]      = new_arg;
    }

    void finalize(context& ctx, const shape&, const std::vector<shape>&)
    {
        if(is_first)
        {
            argument a = to_gpu(l.get_argument());
            arg_map    = store_preallocated_param(ctx, id, a);
            is_first   = false;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const hip_copy_literal& x)
    {
        os << x.name() << "[id=" << x.id << "]";
        return os;
    }

    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        (void)ins;
        return false;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
