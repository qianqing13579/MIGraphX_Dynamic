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
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/convolution.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/add.hpp>
#include <migraphx/gpu/mul.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/device/layernorm.hpp>
#include <migraphx/gpu/device/gelu.hpp>
#include <migraphx/gpu/device/mul_add.hpp>
#include <migraphx/gpu/device/add_clip.hpp>
#include <migraphx/gpu/device/add_relu.hpp>
#include <migraphx/gpu/device/add_sigmoid.hpp>
#include <migraphx/gpu/device/add_tanh.hpp>
#include <migraphx/gpu/device/mul_add_relu.hpp>
#include <migraphx/gpu/device/add.hpp>
#include <migraphx/gpu/device/sigmoid.hpp>
#include <migraphx/gpu/device/div.hpp>
#include <migraphx/gpu/device/pow.hpp>
#include <migraphx/gpu/convert.hpp>
#include <migraphx/gpu/Shape.hpp>
#include <migraphx/gpu/gather.hpp>
#include <migraphx/match/layernorm.hpp>
#include <migraphx/match/gelu_erf.hpp>
#include <migraphx/match/gelu_tanh.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/array.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/make_op.hpp>
#include <cmath>
#include <set>
#include <migraphx/value.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/gpu/compile_gen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MIOPEN_FUSION)

struct fusion
{
    using op_t = miopenFusionOpDescriptor_t;
    shared<fusion_plan_descriptor> fp;

    // Used as a temporary hack to keep descriptor references alive
    std::vector<std::shared_ptr<void>> storage;

    template <class T>
    auto keep_alive(T x)
    {
        auto result = share(std::move(x));
        storage.push_back(result);
        return result;
    }

    fusion() = default;

    fusion(const shape& input)
    {
        assert(input.standard());
        auto t = make_tensor(input);
        fp     = make_fusion_plan(t);
        assert(fp);
        keep_alive(std::move(t));
    }

    bool empty() const { return fp == nullptr; }

    op_t operator[](std::size_t i) const
    {
        assert(fp);
        op_t result;
        auto status = miopenFusionPlanGetOp(fp.get(), i, &result);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Failed retrieving operator at " + std::to_string(i));
        return result;
    }

    auto get() const
    {
        assert(fp);
        return fp.get();
    }

    op_t create_bias(const shape& bias)
    {
        assert(fp);
        op_t result;
        auto b      = shape{bias.type(), {1, bias.lens().at(1), 1, 1}};
        auto t      = keep_alive(make_tensor(b));
        auto status = miopenCreateOpBiasForward(fp.get(), &result, t.get());
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Creating operator failed");
        return result;
    }

    op_t create_relu()
    {
        assert(fp);
        op_t result;
        auto status = miopenCreateOpActivationForward(fp.get(), &result, miopenActivationRELU);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Creating operator failed");
        return result;
    }

    op_t create_conv(const op::convolution& op, const shape& weights)
    {
        assert(fp);
        op_t result;
        auto cd     = keep_alive(make_conv(op));
        auto t      = keep_alive(make_tensor(weights));
        auto status = miopenCreateOpConvForward(fp.get(), &result, cd.get(), t.get());
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Creating operator failed");
        return result;
    }

    shape get_workspace(context&)
    {
        // assert(fp);
        // TODO: Use zero workspace for now
        std::size_t ws_size = 0;
        // int algo_count = 1;
        // miopenConvFwdAlgorithm_t algo;
        // miopenFusionPlanConvolutionGetAlgo(fp.get(), 1, &algo_count, &algo);
        // miopenFusionPlanGetWorkSpaceSize(ctx.get_stream().get_miopen(), fp.get(), &ws_size,
        // algo);
        return shape{shape::int8_type, {ws_size}};
    }

    bool compile(context& ctx)
    {
        assert(fp);
        return miopenCompileFusionPlan(ctx.get_stream().get_miopen(), fp.get()) ==
               miopenStatusSuccess;
    }

    argument execute(context& ctx,
                     const fused_operator_args& fargs,
                     const argument& x,
                     const argument& y) const
    {
        assert(fp);
        auto x_td   = make_tensor(x.get_shape());
        auto y_td   = make_tensor(y.get_shape());
        auto status = miopenExecuteFusionPlan(ctx.get_stream().get_miopen(),
                                              fp.get(),
                                              x_td.get(),
                                              x.implicit(),
                                              y_td.get(),
                                              y.implicit(),
                                              fargs.get());
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Failed to execute fusion plan");
        return y;
    }
};

const std::unordered_set<std::string>& get_supported_archs()
{
    static std::unordered_set<std::string> supported_archs{"gfx900", "gfx906", "gfx908", "gfx1030"};
    return supported_archs;
}

MIGRAPHX_PRED_MATCHER(bias_shape, instruction_ref ins)
{
    auto&& s = ins->get_shape();
    return s.broadcasted() and s.strides().size() == 4 and s.strides()[0] == 0 and
           s.strides()[1] != 0 and s.strides()[2] == 0 and s.strides()[3] == 0;
}

MIGRAPHX_PRED_MATCHER(fusable_conv, instruction_ref ins)
{
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    if(not contains(get_supported_archs(), device_name))
        return false;
    if(enabled(MIGRAPHX_ENABLE_MIOPEN_FUSION{}) == false)
        return false;
    if(ins->name() != "gpu::convolution")
        return false;
    if(ins->get_shape().type() != shape::float_type)
        return false;
    auto wei = ins->inputs().at(1)->get_shape();
    assert(wei.lens().size() == 4);
    auto conv = any_cast<miopen_convolution>(ins->get_operator());
    if(conv.op.group > 1)
        return false;
    if(wei.lens()[1] > 512 and conv.algo != miopenConvolutionFwdAlgoWinograd)
        return false;

    // Do not fuse non-symmetric input
    auto input_lens = ins->inputs().at(0)->get_shape().lens();
    if(input_lens[2] != input_lens[3] or wei.lens()[2] != wei.lens()[3])
        return false;

    auto op = conv.op;
    // Dont fuse winograd for non-3x3s since there is no fused windograd for those configs
    if(conv.algo == miopenConvolutionFwdAlgoWinograd and wei.lens()[2] != 3 and
       wei.lens()[3] != 3 and contains({{1, 1}}, op.stride))
        return false;
    return contains({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}}, op.padding) and
           contains({{0, 0}, {1, 1}}, op.stride) and contains({{1, 1}}, op.dilation);
}

struct hip_triadd : ternary_device<hip_triadd, &device::add>
{
};
MIGRAPHX_REGISTER_OP(hip_triadd)

struct hip_triadd_clip : quinary_device<hip_triadd_clip, &device::add_clip>
{
};
MIGRAPHX_REGISTER_OP(hip_triadd_clip)

struct hip_add_clip : quaternary_device<hip_add_clip, &device::add_clip>
{
};
MIGRAPHX_REGISTER_OP(hip_add_clip)

struct hip_triadd_relu : ternary_device<hip_triadd_relu, &device::add_relu>
{
};
MIGRAPHX_REGISTER_OP(hip_triadd_relu)

struct hip_triadd_sigmoid : ternary_device<hip_triadd_sigmoid, &device::add_sigmoid>
{
};
MIGRAPHX_REGISTER_OP(hip_triadd_sigmoid)

struct hip_triadd_tanh : ternary_device<hip_triadd_tanh, &device::add_tanh>
{
};
MIGRAPHX_REGISTER_OP(hip_triadd_tanh)

struct hip_add_relu : binary_device<hip_add_relu, &device::add_relu>
{
};
MIGRAPHX_REGISTER_OP(hip_add_relu)

struct hip_add_sigmoid : binary_device<hip_add_sigmoid, &device::add_sigmoid>
{
};
MIGRAPHX_REGISTER_OP(hip_add_sigmoid)

struct hip_add_tanh : binary_device<hip_add_tanh, &device::add_tanh>
{
};
MIGRAPHX_REGISTER_OP(hip_add_tanh)

struct hip_layernorm : unary_device<hip_layernorm, &device::layernorm>
{
    // Empty finalize to skip dimension reduction
    void finalize(context&, const shape&, const std::vector<shape>&) {}
};
MIGRAPHX_REGISTER_OP(hip_layernorm)

struct hip_add_layernorm : binary_device<hip_add_layernorm, &device::add_layernorm>
{
    // Empty finalize to skip dimension reduction
    void finalize(context&, const shape&, const std::vector<shape>&) {}
};
MIGRAPHX_REGISTER_OP(hip_add_layernorm)

struct hip_triadd_layernorm : ternary_device<hip_triadd_layernorm, &device::triadd_layernorm>
{
    // Empty finalize to skip dimension reduction
    void finalize(context&, const shape&, const std::vector<shape>&) {}
};
MIGRAPHX_REGISTER_OP(hip_triadd_layernorm)

struct hip_gelu : unary_device<hip_gelu, &device::gelu>
{
};
MIGRAPHX_REGISTER_OP(hip_gelu)

struct hip_add_gelu : binary_device<hip_add_gelu, &device::add_gelu>
{
};
MIGRAPHX_REGISTER_OP(hip_add_gelu)

struct hip_gelu_new : unary_device<hip_gelu_new, &device::gelu_new>
{
};
MIGRAPHX_REGISTER_OP(hip_gelu_new)

struct hip_add_gelu_new : binary_device<hip_add_gelu_new, &device::add_gelu_new>
{
};
MIGRAPHX_REGISTER_OP(hip_add_gelu_new)

struct hip_mul_add : ternary_device<hip_mul_add, &device::mul_add>
{
};
MIGRAPHX_REGISTER_OP(hip_mul_add)

struct hip_mul_add_relu : ternary_device<hip_mul_add_relu, &device::mul_add_relu>
{
};
MIGRAPHX_REGISTER_OP(hip_mul_add_relu)

// silu算子融合
struct hip_sigmoid_mul : unary_device<hip_sigmoid_mul, &device::sigmoid_mul>
{
};
MIGRAPHX_REGISTER_OP(hip_sigmoid_mul)
struct hip_add_sigmoid_mul : binary_device<hip_add_sigmoid_mul, &device::add_sigmoid_mul>
{
};
MIGRAPHX_REGISTER_OP(hip_add_sigmoid_mul)
struct hip_add_sigmoid_mul_add
    : ternary_device<hip_add_sigmoid_mul_add, &device::add_sigmoid_mul_add>
{
};
MIGRAPHX_REGISTER_OP(hip_add_sigmoid_mul_add)
struct hip_sigmoid_mul_add : binary_device<hip_sigmoid_mul_add, &device::sigmoid_mul_add>
{
};
MIGRAPHX_REGISTER_OP(hip_sigmoid_mul_add)

// gpt2中的transformer decoder
struct hip_div_mul_add : quaternary_device<hip_div_mul_add, &device::div_mul_add>
{
};
MIGRAPHX_REGISTER_OP(hip_div_mul_add)
struct hip_mul_add_mul : quaternary_device<hip_mul_add_mul, &device::mul_add_mul>
{
};
MIGRAPHX_REGISTER_OP(hip_mul_add_mul)
struct hip_mul_add_mul_tanh : quaternary_device<hip_mul_add_mul_tanh, &device::mul_add_mul_tanh>
{
};
MIGRAPHX_REGISTER_OP(hip_mul_add_mul_tanh)
struct hip_mul_add_mul_tanh_add
    : quinary_device<hip_mul_add_mul_tanh_add, &device::mul_add_mul_tanh_add>
{
};
MIGRAPHX_REGISTER_OP(hip_mul_add_mul_tanh_add)
struct hip_pow_mul_add_mul_tanh_add
    : six_device<hip_pow_mul_add_mul_tanh_add, &device::pow_mul_add_mul_tanh_add>
{
};
MIGRAPHX_REGISTER_OP(hip_pow_mul_add_mul_tanh_add)
struct hip_pow_mul_add_mul_tanh_add_mul_mul
    : eight_device<hip_pow_mul_add_mul_tanh_add_mul_mul, &device::pow_mul_add_mul_tanh_add_mul_mul>
{
};
MIGRAPHX_REGISTER_OP(hip_pow_mul_add_mul_tanh_add_mul_mul)
struct hip_mul_add_sqrt : ternary_device<hip_mul_add_sqrt, &device::mul_add_sqrt>
{
};
MIGRAPHX_REGISTER_OP(hip_mul_add_sqrt)

struct hip_add_mul : ternary_device<hip_add_mul, &device::add_mul>
{
};
MIGRAPHX_REGISTER_OP(hip_add_mul)
struct hip_add_mul_tanh : ternary_device<hip_add_mul_tanh, &device::add_mul_tanh>
{
};
MIGRAPHX_REGISTER_OP(hip_add_mul_tanh)

// shape->convert->gather中的shape->convert融合
struct hip_shape_convert
{
    literal max_shape; // 最大shape
    shape::type_t target_type = shape::float_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.target_type, "target_type"), f(self.max_shape, "max_shape"));
    }

    std::string name() const { return "gpu::shape_convert"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);

        // 首先获取shape计算结果，然后转换为float类型
        std::vector<std::size_t> arg_shape = inputs[0].lens();
        shape output_shape_of_shape_op{migraphx::shape::float_type, {arg_shape.size()}};

        return output_shape_of_shape_op;
    }

    argument compute(context& ctx, const shape& output_shape, std::vector<argument> args) const
    {
        // 首先执行shape算子的计算，并转换为float类型
        std::vector<std::size_t> arg_shape = args.front().get_shape().lens();
        migraphx::shape s(migraphx::shape::float_type, {arg_shape.size()});
        migraphx::argument data{s};
        std::transform(arg_shape.begin(), arg_shape.end(), (float*)data.data(), [](auto i) {
            return float(i);
        });

        return data;
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        return false;
    }
};
MIGRAPHX_REGISTER_OP(hip_shape_convert)

// shape->convert融合，hip_shape_convert2与hip_shape_convert的区别是hip_shape_convert返回的是cpu数据，因为后面有gather算子
struct hip_shape_convert2
{
    literal max_shape; // 最大shape
    shape::type_t target_type = shape::float_type;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.target_type, "target_type"), f(self.max_shape, "max_shape"));
    }

    std::string name() const { return "gpu::shape_convert2"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);

        // 首先获取shape计算结果，然后转换为float类型
        std::vector<std::size_t> arg_shape = inputs[0].lens();
        shape output_shape_of_shape_op{migraphx::shape::float_type, {arg_shape.size()}};

        return output_shape_of_shape_op;
    }

    argument compute(context& ctx, const shape& output_shape, std::vector<argument> args) const
    {
        // 首先执行shape算子的计算，并转换为float类型
        std::vector<std::size_t> arg_shape = args.front().get_shape().lens();
        migraphx::shape s{migraphx::shape::float_type, {arg_shape.size()}};
        migraphx::argument data{s};
        std::transform(arg_shape.begin(), arg_shape.end(), (float*)data.data(), [](auto i) {
            return float(i);
        });

        // 拷贝到gpu
        hipMemcpyAsync(args.back().data(),
                       data.data(),
                       data.get_shape().bytes(),
                       hipMemcpyHostToDevice,
                       ctx.get_stream().get());

        return args.back();
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        return false;
    }
};
MIGRAPHX_REGISTER_OP(hip_shape_convert2)

// 该算子用来等价替换shape->gather模式中的gather算子
struct hip_gather_for_shape
{
    int64_t axis = 0;
    std::vector<int> index;
    bool is_first = true;
    argument result;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), f(self.index, "index"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{op::normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "gpu::gather_for_shape"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);

        auto lens = inputs[0].lens();
        auto type = inputs[0].type();
        lens.erase(lens.begin() + axis);
        if(not inputs[1].scalar())
        {
            auto ind_lens = inputs[1].lens();
            lens.insert(lens.begin() + axis, ind_lens.begin(), ind_lens.end());
        }

        // for scalar output
        if(lens.empty())
        {
            return {type};
        }

        return {type, lens};
    }

    argument compute(context& ctx, const shape& output_shape, std::vector<argument> args) const
    {
        // 计算gather
        for(int i = 0; i < index.size(); ++i)
        {
            ((float*)result.data())[i] = ((float*)args[0].data())[index[i]];
        }

        // 将计算结果拷贝到gpu
        hipMemcpyAsync(args.back().data(),
                       result.data(),
                       result.get_shape().bytes(),
                       hipMemcpyHostToDevice,
                       ctx.get_stream().get());

        return args.back();
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    void finalize(context& ctx, const shape& output_shape, std::vector<shape> inputs)
    {
        if(is_first)
        {
            result   = argument{output_shape};
            is_first = false;
        }
    }

    bool do_reshape(instruction_ref ins, std::unordered_map<instruction_ref, argument>& results)
    {
        return false;
    }
};
MIGRAPHX_REGISTER_OP(hip_gather_for_shape)

void move_broadcasted_back(std::vector<instruction_ref>& args)
{
    // Ensure the last arguments is the broadcasted one
    auto last = std::prev(args.end());
    auto it =
        std::find_if(args.begin(), last, [](auto arg) { return arg->get_shape().broadcasted(); });
    if(it != last)
        std::swap(*it, *std::prev(last));
}

void move_standard_front(std::vector<instruction_ref>& args)
{
    // Ensure the first arguments is the standard one
    auto last = std::prev(args.end());
    auto it =
        std::find_if(args.begin(), last, [](auto arg) { return arg->get_shape().standard(); });
    if(it != last)
        std::swap(*it, args.front());
}

auto gpu_name(const std::string& s) { return match::name("gpu::" + s); }

namespace {

template <class... Strings>
inline auto precompile_name(Strings... names) // NOLINT
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(ins->name() != "gpu::precompile_op")
            return false;
        auto op = from_value<operation>(ins->get_operator().to_value().at("op"));
        return (contains({names...}, op.name()));
    });
}

struct find_layernorm
{
    auto matcher() const { return match::name("gpu::prelayernorm"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto layernorm_ins = r.result;
        auto x_ins         = layernorm_ins->inputs()[0];

        // 分配内存
        auto output = m.insert_instruction(
            layernorm_ins, make_op("allocate", {{"shape", to_value(layernorm_ins->get_shape())}}));

        m.replace_instruction(layernorm_ins, hip_layernorm{}, x_ins, output);
    }
};

struct find_triadd_layernorm
{
    auto matcher() const
    {
        return match::name("gpu::layernorm")(
            match::arg(0)(match::name("gpu::triadd")(match::used_once())));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto triadd = ins->inputs().front();
        m.replace_instruction(ins, hip_triadd_layernorm{}, triadd->inputs());
    }
};

struct find_add_layernorm
{
    auto matcher() const
    {
        return match::name("gpu::layernorm")(
            match::arg(0)(match::name("gpu::add")(match::used_once())));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto add = ins->inputs().front();
        m.replace_instruction(ins, hip_add_layernorm{}, add->inputs());
    }
};

struct find_pointwise_layernorm
{
    auto matcher() const
    {
        return match::name("gpu::layernorm")(
            match::arg(0)(precompile_name("pointwise")(match::used_once())));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto layernorm_ins                = r.result;
        instruction_ref pointwise_ins     = layernorm_ins->inputs()[0];
        std::vector<instruction_ref> args = pointwise_ins->inputs();

        auto* pm         = pointwise_ins->module_inputs().front();
        auto kernel_name = gpu::gen::generate_name_from_ops(*pm) + "_kernel";

        if(kernel_name == "add_add_kernel")
        {
            args.back() = layernorm_ins->inputs().back();
            m.replace_instruction(layernorm_ins, hip_triadd_layernorm{}, args);
        }
        else if(kernel_name == "add_kernel")
        {
            args.back() = layernorm_ins->inputs().back();
            m.replace_instruction(layernorm_ins, hip_add_layernorm{}, args);
        }
    }
};

struct find_gelu
{
    auto erf_fn() const
    {
        auto mul_1_sqrt_2 = match::name("gpu::mul")(
            match::either_arg(0, 1)(match::any().bind("x"), match::has_value(M_SQRT1_2, 1e-3)));
        auto div_sqrt_2 = match::name("gpu::div")(
            match::either_arg(0, 1)(match::any().bind("x"), match::has_value(M_SQRT2, 1e-3)));
        return match::name("gpu::erf")(
            match::used_once(),
            match::arg(0)(match::used_once(), match::any_of(mul_1_sqrt_2, div_sqrt_2)));
    }

    auto matcher() const
    {

        return match::name("gpu::mul")(match::either_arg(0, 1)(
            match::name("gpu::mul")(
                match::used_once(),
                match::either_arg(0, 1)(
                    match::name("gpu::add")(
                        match::used_once(),
                        match::either_arg(0, 1)(erf_fn(), match::has_value(1.0f, 1e-3))),
                    match::any().bind("y"))),
            match::has_value(0.5f, 1e-3)

                ));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto y_ins = r.instructions["y"];
        auto args  = ins->inputs();
        if(x_ins != y_ins)
        {
            return;
        }

        m.replace_instruction(ins, hip_gelu{}, x_ins, args.back());
    }
};

struct find_add_gelu
{
    auto matcher() const
    {
        return match::name("gpu::gelu")(match::arg(0)(match::name("gpu::add").bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto add_ins = r.instructions["add"];
        auto ins     = r.result;
        auto args    = add_ins->inputs();
        move_standard_front(args);
        move_broadcasted_back(args);

        // 如果standard的是一个常量，则需要调整参数顺序
        if(args[0]->name() == "@literal")
        {
            std::swap(args[0], args[1]);
        }

        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_add_gelu{}, args);
    }
};

struct find_gelu_new
{
    bool fast_math = false;

    auto matcher() const
    {
        return match::name("gpu::mul")(match::either_arg(0, 1)(
            match::name("gpu::mul")(
                match::used_once(),
                match::either_arg(0, 1)(match::has_value(0.5f, 1e-3), match::any().bind("x"))),
            match::name("gpu::add")(match::either_arg(0, 1)(
                match::has_value(1.0f, 1e-3),
                match::name("gpu::tanh")(
                    match::used_once(),
                    match::arg(0)(match::name("gpu::mul")(
                        match::used_once(),
                        match::either_arg(0,
                                          1)(match::has_value(0.7978f, 1e-3),
                                             match::name("gpu::add")(
                                                 match::used_once(),
                                                 match::either_arg(0, 1)(
                                                     match::name("gpu::mul")(
                                                         match::used_once(),
                                                         match::either_arg(0, 1)(
                                                             match::has_value(0.0447f, 1e-3),
                                                             match::name("gpu::pow")(
                                                                 match::used_once(),
                                                                 match::either_arg(0, 1)(
                                                                     match::has_value(3.0f, 1e-3),
                                                                     match::any().bind("z"))))),
                                                     match::any().bind("y")))))))))

                ));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto y_ins = r.instructions["y"];
        auto z_ins = r.instructions["z"];
        auto args  = ins->inputs();

        if((x_ins->get_shape() == y_ins->get_shape()) && (y_ins->get_shape() == z_ins->get_shape()))
        {
            m.replace_instruction(ins, hip_gelu_new{}, x_ins, args.back());
        }
    }
};

struct find_add_gelu_new
{
    auto matcher() const
    {
        return match::name("gpu::gelu_new")(match::arg(0)(match::name("gpu::add").bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto add_ins = r.instructions["add"];
        auto ins     = r.result;
        auto args    = add_ins->inputs();
        move_standard_front(args);
        move_broadcasted_back(args);

        // 如果standard的是一个常量，则需要调整参数顺序
        if(args[0]->name() == "@literal")
        {
            std::swap(args[0], args[1]);
        }

        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_add_gelu_new{}, args);
    }
};

struct find_add_clip
{
    auto matcher() const
    {
        return match::name(std::unordered_set<std::string>{"gpu::clip", "gpu::clipped_relu"})(
            match::arg(0)(match::any_of(match::name("gpu::add"),
                                        match::name("gpu::triadd"),
                                        match::any_of[match::inputs()](match::standard_shape()))
                              .bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto add_ins  = r.instructions["add"];
        auto ins      = r.result;
        auto ins_args = ins->inputs();
        auto add_args = add_ins->inputs();
        move_standard_front(add_args);
        move_broadcasted_back(add_args);

        // 如果standard的是一个常量，则需要调整参数顺序
        if(add_args[0]->name() == "@literal")
        {
            std::swap(add_args[0], add_args[1]);
        }

        // Use the allocation from the clip operator
        add_args.pop_back();
        add_args.insert(add_args.end(), std::next(ins_args.begin()), ins_args.end());
        if(add_ins->name() == "gpu::add")
            m.replace_instruction(ins, hip_add_clip{}, add_args);
        else if(add_ins->name() == "gpu::triadd")
            m.replace_instruction(ins, hip_triadd_clip{}, add_args);
    }
};

struct find_add_unary
{
    std::string op_name;
    operation binary_add_op;
    operation ternary_add_op;
    auto matcher() const
    {
        return match::name(op_name)(match::arg(0)(
            match::used_once(),
            match::any_of(match::name("gpu::add"),
                          match::name("gpu::triadd"),
                          match::any_of(match::name("@literal"),
                                        match::any_of[match::inputs()](match::standard_shape())))
                .bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto add_ins = r.instructions["add"];
        auto ins     = r.result;
        auto args    = add_ins->inputs();
        move_standard_front(args);
        move_broadcasted_back(args);

        // 如果standard的是一个常量，则需要调整参数顺序
        if(args[0]->name() == "@literal")
        {
            std::swap(args[0], args[1]);
        }

        // Use the allocation from the relu operator
        args.back() = ins->inputs().back();
        if(add_ins->name() == "gpu::add")
            m.replace_instruction(ins, binary_add_op, args);
        else if(add_ins->name() == "gpu::triadd")
            m.replace_instruction(ins, ternary_add_op, args);
    }
};

// silu算子融合
struct find_sigmoid_mul
{
    auto matcher() const
    {
        return match::name("gpu::mul")(
            match::either_arg(0, 1)(match::name("gpu::sigmoid")(match::used_once()).bind("sigmoid"),
                                    match::any().bind("b")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto sigmoid_ins = r.instructions["sigmoid"];
        auto b_ins       = r.instructions["b"];
        auto ins         = r.result;
        auto args        = sigmoid_ins->inputs();
        if(args[0] == b_ins)
        {
            args.back() = ins->inputs().back();
            m.replace_instruction(ins, hip_sigmoid_mul{}, args);
        }
    }
};

struct find_add_sigmoid_mul
{
    auto matcher() const
    {
        return match::name("gpu::sigmoid_mul")(
            match::arg(0)(match::name("gpu::add")(match::used_once()).bind("add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto add_ins = r.instructions["add"];
        auto ins     = r.result;
        auto args    = add_ins->inputs();
        args.back()  = ins->inputs().back();

        m.replace_instruction(ins, hip_add_sigmoid_mul{}, args);
    }
};

struct find_add_sigmoid_mul_add
{
    auto matcher() const
    {
        return match::name("gpu::add")(match::either_arg(0, 1)(
            match::name("gpu::add_sigmoid_mul")(match::used_once()).bind("add_sigmoid_mul"),
            match::any().bind("b")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto add_sigmoid_mul_ins = r.instructions["add_sigmoid_mul"];
        auto b_ins               = r.instructions["b"];
        auto ins                 = r.result;
        auto ins_args            = ins->inputs();
        auto args                = add_sigmoid_mul_ins->inputs();

        ins_args[1] = args[0];
        ins_args.insert(std::prev(ins_args.end()), args[1]);

        m.replace_instruction(ins, hip_add_sigmoid_mul_add{}, ins_args);
    }
};

// gpt2中的transformer decoder
struct find_div_mul_add
{
    auto matcher() const
    {
        return match::name("gpu::mul_add")(
            match::arg(0)(match::name("gpu::div")(match::used_once()).bind("div")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto div_ins = r.instructions["div"];
        auto ins     = r.result;
        auto args    = div_ins->inputs();

        // 如果输入参数都是广播，则不发生融合，原因：src/targets/gpu/include/migraphx/gpu/oper.hpp中do_reshape()中出现的原因
        auto is_broadcasted = [](auto arg) { return arg->get_shape().broadcasted(); };
        if(std::count_if(args.begin(), args.end(), is_broadcasted) >= 2)
            return;

        args.pop_back();
        for(int i = 1; i < ins->inputs().size(); ++i)
        {
            args.push_back(ins->inputs()[i]);
        }
        m.replace_instruction(ins, hip_div_mul_add{}, args);
    }
};
struct find_mul_add_mul
{
    auto matcher() const
    {
        return match::name("gpu::mul")(
            match::either_arg(0, 1)(match::name("gpu::mul_add")(match::used_once()).bind("mul_add"),
                                    match::any().bind("b")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto mul_add_ins = r.instructions["mul_add"];
        auto b_ins       = r.instructions["b"];
        auto ins         = r.result;
        auto args        = mul_add_ins->inputs();

        args.insert(std::prev(args.end()), b_ins);
        args.back() = ins->inputs().back();

        m.replace_instruction(ins, hip_mul_add_mul{}, args);
    }
};

struct find_mul_add_sqrt
{
    auto matcher() const
    {
        return match::name("gpu::sqrt")(
            match::arg(0)(match::name("gpu::mul_add")(match::used_once()).bind("mul_add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto mul_add_ins = r.instructions["mul_add"];
        auto ins         = r.result;
        auto args        = mul_add_ins->inputs();

        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_mul_add_sqrt{}, args);
    }
};

struct find_mul_add_mul_tanh
{
    auto matcher() const
    {
        return match::name("gpu::tanh")(
            match::arg(0)(match::name("gpu::mul_add_mul")(match::used_once()).bind("mul_add_mul")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto mul_add_ins = r.instructions["mul_add_mul"];
        auto ins         = r.result;
        auto args        = mul_add_ins->inputs();

        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_mul_add_mul_tanh{}, args);
    }
};
struct find_mul_add_mul_tanh_add
{
    auto matcher() const
    {
        return match::name("gpu::add")(match::either_arg(0, 1)(
            match::name("gpu::mul_add_mul_tanh")(match::used_once()).bind("mul_add_mul_tanh"),
            match::any().bind("b")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto mul_add_mul_ins = r.instructions["mul_add_mul_tanh"];
        auto b_ins           = r.instructions["b"];
        auto ins             = r.result;
        auto args            = mul_add_mul_ins->inputs();

        args.insert(std::prev(args.end()), b_ins);
        args.back() = ins->inputs().back();

        m.replace_instruction(ins, hip_mul_add_mul_tanh_add{}, args);
    }
};

// shape->convert->gather
struct find_shape_convert
{
    auto matcher() const
    {
        return match::name("gpu::convert")(
            match::arg(0)(match::name("gpu::Shape")(match::used_once())));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;

        bool all_gather = std::all_of(ins->outputs().begin(), ins->outputs().end(), [](auto ins) {
            return ins->name() == "gpu::gather";
        });

        // 如果输出都是gather，则融合
        if(all_gather)
        {
            auto ins_arg = ins->inputs();

            ins_arg[0] = ins->inputs()[0]->inputs()[0];

            shape::type_t target_type =
                any_cast<gpu::hip_convert>(ins->get_operator()).op.target_type;
            literal max_shape =
                any_cast<gpu::hip_Shape>(ins->inputs()[0]->get_operator()).op.max_shape;

            m.replace_instruction(ins, hip_shape_convert{max_shape, target_type}, ins_arg);
        }
    }
};

// shape->convert通用模式
struct find_shape_convert2
{
    auto matcher() const
    {
        return match::name("gpu::convert")(
            match::arg(0)(match::name("gpu::Shape")(match::used_once())));
    }

    void apply(module& m, const match::matcher_result& r) const
    {

        auto ins       = r.result;
        auto shape_ins = ins->inputs()[0];
        auto args      = shape_ins->inputs();

        args.back() = ins->inputs().back();

        shape::type_t target_type = any_cast<gpu::hip_convert>(ins->get_operator()).op.target_type;
        literal max_shape = any_cast<gpu::hip_Shape>(ins->inputs()[0]->get_operator()).op.max_shape;
        m.replace_instruction(ins, hip_shape_convert2{max_shape, target_type}, args);
    }
};

struct find_gather_for_shape
{
    auto matcher() const
    {
        return match::name("gpu::gather")(
            match::either_arg(0, 1)(match::name("gpu::shape_convert"), match::any().bind("b")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto ins_arg = ins->inputs();

        int64_t axis = any_cast<gpu::hip_gather>(ins->get_operator()).op.axis;

        // 获取gather的索引
        argument index = ins_arg[1]->eval();
        std::vector<int> index2;
        for(int i = 0; i < index.get_shape().elements(); ++i)
        {
            index2.push_back(int(((float*)index.data())[i]));
        }

        // 删除第二个参数
        ins_arg.erase(ins_arg.begin() + 1);

        m.replace_instruction(ins, hip_gather_for_shape{axis, index2}, ins_arg);
    }
};

struct find_pow_mul_add_mul_tanh_add
{
    auto matcher() const
    {
        return match::name("gpu::mul_add_mul_tanh_add")(
            match::nargs(6),
            match::either_arg(0, 1)(match::name("gpu::pow")(match::used_once()).bind("pow"),
                                    match::any().bind("b")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto pow_ins = r.instructions["pow"];
        auto ins     = r.result;
        auto args    = pow_ins->inputs();

        for(int i = 1; i <= ins->inputs().size() - 2; ++i)
        {
            args.insert(std::prev(args.end()), ins->inputs()[i]);
        }

        args.back() = ins->inputs().back();

        m.replace_instruction(ins, hip_pow_mul_add_mul_tanh_add{}, args);
    }
};

struct find_pow_mul_add_mul_tanh_add_mul_mul
{
    auto matcher() const
    {
        return match::name("gpu::mul")(
            match::either_arg(0, 1)(match::name("gpu::pow_mul_add_mul_tanh_add")(match::used_once())
                                        .bind("pow_mul_add_mul_tanh_add"),
                                    match::name("gpu::mul")(match::used_once()).bind("mul")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto mul_ins                       = r.instructions["mul"];
        auto mul_args                      = mul_ins->inputs();
        auto pow_mul_add_mul_tanh_add_ins  = r.instructions["pow_mul_add_mul_tanh_add"];
        auto pow_mul_add_mul_tanh_add_args = pow_mul_add_mul_tanh_add_ins->inputs();
        auto ins                           = r.result;

        for(int i = 0; i <= mul_args.size() - 2; ++i)
        {
            pow_mul_add_mul_tanh_add_args.insert(std::prev(pow_mul_add_mul_tanh_add_args.end()),
                                                 mul_args[i]);
        }

        pow_mul_add_mul_tanh_add_args.back() = ins->inputs().back();

        m.replace_instruction(
            ins, hip_pow_mul_add_mul_tanh_add_mul_mul{}, pow_mul_add_mul_tanh_add_args);
    }
};
struct find_add_mul
{
    auto matcher() const
    {
        return match::name("gpu::mul")(match::either_arg(0, 1)(
            match::name("gpu::add")(match::used_once()).bind("add"), match::any().bind("b")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto add_ins = r.instructions["add"];
        auto b_ins   = r.instructions["b"];
        auto ins     = r.result;
        auto args    = add_ins->inputs();

        move_standard_front(args);
        move_broadcasted_back(args);

        // 如果standard的是一个常量，则需要调整参数顺序
        if(args[0]->name() == "@literal")
        {
            std::swap(args[0], args[1]);
        }

        args.insert(std::prev(args.end()), b_ins);

        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_add_mul{}, args);
    }
};
struct find_add_mul_tanh
{
    auto matcher() const
    {
        return match::name("gpu::tanh")(
            match::arg(0)(match::name("gpu::add_mul")(match::used_once()).bind("add_mul")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto mul_add_ins = r.instructions["add_mul"];
        auto ins         = r.result;
        auto args        = mul_add_ins->inputs();

        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_add_mul_tanh{}, args);
    }
};
/////////////////////gpt2算子融合/////////////

struct find_triadd
{
    auto matcher() const
    {
        return match::name("gpu::add")(match::either_arg(0, 1)(
            match::name("gpu::add")(match::used_once()).bind("add"),
            match::any(match::any_of(match::name("@literal"),
                                     match::any_of[match::inputs()](match::standard_shape())))
                .bind("input")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto add_ins   = r.instructions["add"];
        auto input_ins = r.instructions["input"];
        auto ins       = r.result;
        auto args      = add_ins->inputs();

        // 如果输入参数都是广播，则不发生融合，原因：src/targets/gpu/include/migraphx/gpu/oper.hpp中do_reshape()中出现的原因
        auto is_broadcasted = [](auto arg) { return arg->get_shape().broadcasted(); };
        if(std::count_if(args.begin(), args.end(), is_broadcasted) >= 2)
            return;

        args.insert(args.begin(), input_ins);
        move_standard_front(args);
        move_broadcasted_back(args);

        // 如果standard的是一个常量，则需要调整参数顺序
        if(args[0]->name() == "@literal")
        {
            std::swap(args[0], args[1]);
        }

        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_triadd{}, args);
    }
};

struct find_mul_add
{
    auto matcher() const
    {
        return match::name("gpu::add")(match::either_arg(0, 1)(
            match::name("gpu::mul")(match::used_once()).bind("mul"), match::any().bind("b")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto mul_ins = r.instructions["mul"];
        auto b_ins   = r.instructions["b"];
        auto ins     = r.result;
        auto args    = mul_ins->inputs();
        assert(mul_ins != b_ins);

        // 如果输入参数都是广播，则不发生融合，原因：src/targets/gpu/include/migraphx/gpu/oper.hpp中do_reshape()中出现的原因
        auto is_broadcasted = [](auto arg) { return arg->get_shape().broadcasted(); };
        if(std::count_if(args.begin(), args.end(), is_broadcasted) >= 2)
            return;

        move_standard_front(args);
        move_broadcasted_back(args);

        // 如果standard的是一个常量，则需要调整参数顺序
        if(args[0]->name() == "@literal")
        {
            std::swap(args[0], args[1]);
        }

        args.insert(std::prev(args.end()), b_ins);

        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_mul_add{}, args);
    }
};

struct find_mul_add_relu
{
    auto matcher() const
    {
        return match::name("gpu::relu")(
            match::arg(0)(match::name("gpu::mul_add")(match::used_once()).bind("mul_add")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto mul_add_ins = r.instructions["mul_add"];
        auto ins         = r.result;
        auto args        = mul_add_ins->inputs();

        // Use the allocation from the relu operator
        args.back() = ins->inputs().back();
        m.replace_instruction(ins, hip_mul_add_relu{}, args);
    }
};

struct miopen_fusion
{
    struct fuse_op_data
    {
        operation op;
        float alpha = 1;
        float beta  = 0;
    };
    struct fuse_op : fuse_op_data, reflect_equality<fuse_op>, reflect_stream<fuse_op>
    {
        template <class Self, class F>
        static auto reflect(Self& self, F f)
        {
            return pack(f(self.op, "op"), f(self.alpha, "alpha"), f(self.beta, "beta"));
        }
    };
    std::vector<fuse_op> ops = {};
    fusion f                 = {};
    std::function<void(context&, const fusion&, const std::vector<argument>&)> execute;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.ops, "ops"));
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    value compile(context& ctx, const shape&, std::vector<shape> inputs)
    {
        // Compensate for allocation
        inputs.pop_back();
        std::size_t i = 0;
        f             = fusion(inputs[i]);
        i++;
        std::vector<std::function<void(const fused_operator_args&, const std::vector<argument>&)>>
            invokers;
        for(auto&& fop : ops)
        {
            if(i > inputs.size())
            {
                f = {};
                return {};
            }
            if(fop.op.name() == "convolution")
            {
                auto* mop = f.create_conv(any_cast<op::convolution>(fop.op), inputs[i]);
                invokers.push_back(
                    [=](const fused_operator_args& fargs, const std::vector<argument>& args) {
                        miopenSetOpArgsConvForward(
                            fargs.get(), mop, &fop.alpha, &fop.beta, args[i].implicit());
                    });
                i++;
            }
            else if(fop.op.name() == "add")
            {
                auto* mop = f.create_bias(inputs[i]);
                invokers.push_back(
                    [=](const fused_operator_args& fargs, const std::vector<argument>& args) {
                        miopenSetOpArgsBiasForward(
                            fargs.get(), mop, &fop.alpha, &fop.beta, args[i].implicit());
                    });
                i++;
            }
            else if(fop.op.name() == "relu")
            {
                auto* mop = f.create_relu();
                invokers.push_back([=](const fused_operator_args& fargs,
                                       const std::vector<argument>&) {
                    miopenSetOpArgsActivForward(fargs.get(), mop, &fop.alpha, &fop.beta, 0, 0, 0);
                });
            }
            else
            {
                f = {};
                return {};
            }
        }
        if(not f.compile(ctx))
        {
            f = {};
            return {};
        }
        execute = [invokers](context& c, const fusion& ff, const std::vector<argument>& args) {
            auto fargs = make_fused_args();
            for(auto&& invoker : invokers)
                invoker(fargs, args);
            ff.execute(c, fargs, args.front(), args.back());
        };
        return {{"workspace", f.get_workspace(ctx).bytes()}};
    }
    void finalize(context& ctx, const shape& output_shape, const std::vector<shape>& inputs)
    {
        // 在动态shape中，需要重新编译，所以需要注释掉
        // if(not f.empty())
        //     return;
        auto v = compile(ctx, output_shape, inputs);
        if(not v.is_object())
            MIGRAPHX_THROW("Failed to compile fusion plan");
    }
    std::string name() const { return "gpu::miopen_fusion"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        if(ops.empty())
            return {};
        // TODO: Check number of arguments
        return ops.front().op.compute_shape({inputs[0], inputs[1]});
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        execute(ctx, f, args);
        return args.back();
    }
};
MIGRAPHX_REGISTER_OP(miopen_fusion)

struct miopen_conv_bias
{
    op::convolution op;
    fusion fp         = {};
    fusion::op_t conv = {};
    fusion::op_t bias = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return op::convolution::reflect(self.op, f);
    }

    std::string name() const { return "gpu::conv_bias"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        // TODO: Check slices
        return op.normalize_compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        auto fargs  = make_fused_args();
        float alpha = 1;
        float beta  = 0;
        miopenSetOpArgsConvForward(fargs.get(), conv, &alpha, &beta, args[1].implicit());
        miopenSetOpArgsBiasForward(fargs.get(), bias, &alpha, &beta, args[3].implicit());
        return fp.execute(ctx, fargs, args[0], args[4]);
    }

    void finalize(context& ctx, const shape&, const std::vector<shape>& inputs)
    {
        fp   = fusion(inputs[0]);
        conv = fp.create_conv(op, inputs[1]);
        bias = fp.create_bias(inputs[3]);
        if(not fp.compile(ctx))
            MIGRAPHX_THROW("Failed to compile fusion plan");
    }

    shape get_workspace(context& ctx) { return fp.get_workspace(ctx); }
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};
MIGRAPHX_REGISTER_OP(miopen_conv_bias)

struct miopen_conv_bias_relu
{
    op::convolution op;
    fusion fp         = {};
    fusion::op_t conv = {};
    fusion::op_t bias = {};
    fusion::op_t relu = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return op::convolution::reflect(self.op, f);
    }

    std::string name() const { return "gpu::conv_bias_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        // TODO: Check slices
        return op.normalize_compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        auto fargs  = make_fused_args();
        float alpha = 1;
        float beta  = 0;
        miopenSetOpArgsConvForward(fargs.get(), conv, &alpha, &beta, args[1].implicit());
        miopenSetOpArgsBiasForward(fargs.get(), bias, &alpha, &beta, args[3].implicit());
        miopenSetOpArgsActivForward(fargs.get(), relu, &alpha, &beta, 0, 0, 0);
        return fp.execute(ctx, fargs, args[0], args[4]);
    }
    void finalize(context& ctx, const shape&, const std::vector<shape>& inputs)
    {
        fp   = fusion(inputs[0]);
        conv = fp.create_conv(op, inputs[1]);
        bias = fp.create_bias(inputs[3]);
        relu = fp.create_relu();
        fp.compile(ctx);
    }

    shape get_workspace(context& ctx) { return fp.get_workspace(ctx); }
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};
MIGRAPHX_REGISTER_OP(miopen_conv_bias_relu)

template <class... Ms>
auto conv_bias(Ms... ms)
{
    return match::name("gpu::add")(
        match::either_arg(0, 1)(bias_shape(match::used_once()).bind("bias"),
                                fusable_conv(match::used_once()).bind("conv")),
        ms...);
}

template <class Op>
void apply_conv_bias(context& ctx, module& m, const match::matcher_result& r)
{
    auto conv_ins    = r.instructions["conv"];
    auto bias_ins    = r.instructions["bias"];
    auto ins         = r.result;
    auto input_ins   = conv_ins->inputs().at(0);
    auto weights_ins = conv_ins->inputs().at(1);
    auto conv_op     = any_cast<miopen_convolution>(conv_ins->get_operator()).op;
    auto alloc_ins   = ins->inputs().back();
    auto old_ws_ins  = conv_ins->inputs().at(2);

    Op cb{conv_op};
    // TODO: Insert ws allocation
    auto ws = cb.get_workspace(ctx);
    (void)ws;
    m.replace_instruction(ins, cb, input_ins, weights_ins, old_ws_ins, bias_ins, alloc_ins);
}

struct find_conv_bias
{
    context* ctx = nullptr;
    auto matcher() const
    {
        return conv_bias(match::none_of(
            match::output(match::name(std::unordered_set<std::string>{"gpu::relu"}))));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        apply_conv_bias<miopen_conv_bias>(*ctx, m, r);
    }
};

struct find_conv_bias_relu
{
    context* ctx = nullptr;
    auto matcher() const { return match::name("gpu::relu")(match::arg(0)(conv_bias())); }

    void apply(module& m, const match::matcher_result& r) const
    {
        apply_conv_bias<miopen_conv_bias_relu>(*ctx, m, r);
    }
};

struct find_conv_pointwise
{
    context* ctx = nullptr;
    auto matcher() const
    {
        return precompile_name("pointwise")(
            match::nargs(3),
            match::either_arg(0, 1)(bias_shape(match::used_once()).bind("bias"),
                                    fusable_conv(match::used_once()).bind("conv")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto conv_ins    = r.instructions["conv"];
        auto bias_ins    = r.instructions["bias"];
        auto ins         = r.result;
        auto input_ins   = conv_ins->inputs().at(0);
        auto weights_ins = conv_ins->inputs().at(1);
        auto conv_op     = any_cast<miopen_convolution>(conv_ins->get_operator()).op;
        auto alloc_ins   = ins->inputs().back();

        module_ref pm = ins->module_inputs().front();

        miopen_fusion op{};
        op.ops.push_back({{conv_op}});
        for(auto&& i : *pm)
        {
            if(i.name()[0] == '@')
                continue;
            op.ops.push_back({{i.get_operator()}});
        }
        std::vector<instruction_ref> inputs = {input_ins, weights_ins, bias_ins, alloc_ins};
        auto v                              = op.compile(*ctx, ins->get_shape(), to_shapes(inputs));
        if(not v.is_object())
            return;
        m.replace_instruction(ins, op, inputs);
    }
};

struct find_gemm_add
{
    auto matcher() const
    {
        return match::name("gpu::add")(
            match::all_of[match::inputs()](match::standard_shape()),
            match::either_arg(0, 1)(match::used_once().bind("c"),
                                    match::name("gpu::gemm")(match::nargs(3)).bind("gemm")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["gemm"];
        auto c_ins    = r.instructions["c"];

        auto gemm = any_cast<rocblas_gemm<op::dot>>(gemm_ins->get_operator());

        // Already fused gemm
        if(not float_equal(gemm.beta, 0))
            return;

        auto inputs = gemm_ins->inputs();
        inputs.pop_back();

        inputs.push_back(c_ins);
        inputs.push_back(ins->inputs().back());

        gemm.beta = 1;
        m.replace_instruction(ins, gemm, inputs);
    }
};

struct find_gemm_pointwise
{
    auto matcher() const
    {
        return precompile_name("pointwise")(
            match::nargs(3),
            match::either_arg(0, 1)(
                match::any_of(match::standard_shape(), match::is_constant()).bind("c"),
                match::name("gpu::gemm")(match::nargs(3), match::used_once()).bind("gemm")));
    }

    // TODO: Move to matcher.hpp
    static auto match_param(const std::string& name)
    {
        return match::make_basic_pred_matcher([=](auto ins) {
            if(ins->name() != "@param")
                return false;
            auto p = any_cast<builtin::param>(ins->get_operator());
            return p.parameter == name;
        });
    }

    template <class M>
    static auto match_mul_const(M m, const std::string& var)
    {
        return match::name("mul")(match::either_arg(0, 1)(match::name("@literal").bind(var), m))
            .bind(var + "_mul");
    }

    static auto match_add(const std::string& input, const std::string& output)
    {
        auto param     = match::name("@param");
        auto add       = match::name("add")(match::args(param, param));
        auto inner_mul = match::any_of(match_mul_const(match_param(input), "alpha"),
                                       match_mul_const(match_param(output), "beta"));
        auto mul_add   = match::name("add")(match::either_arg(0, 1)(inner_mul, param));
        auto add_mul   = match_mul_const(add, "gamma");
        return match::name("@return")(match::args(match::any_of(add, mul_add, add_mul)));
    }

    static float get_float(instruction_ref ins) { return ins->get_literal().at<float>(); }

    template <class Gemm>
    static bool update_gemm(Gemm& gemm, module_ref pm, unsigned input)
    {
        auto names = pm->get_parameter_names();
        if(names.size() != 2)
            return false;
        std::sort(names.begin(), names.end());
        unsigned output = input == 0 ? 1 : 0;
        auto mr         = match::match_instruction(
            *pm, std::prev(pm->end()), match_add(names[input], names[output]));
        if(mr.result == pm->end())
            return false;
        if(contains(mr.instructions, "alpha_mul"))
            gemm.alpha *= get_float(mr.instructions["alpha"]);
        else if(contains(mr.instructions, "beta_mul"))
            gemm.beta *= get_float(mr.instructions["beta"]);
        else if(contains(mr.instructions, "gamma_mul"))
        {
            gemm.alpha *= get_float(mr.instructions["gamma"]);
            gemm.beta *= get_float(mr.instructions["gamma"]);
        }
        return true;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["gemm"];
        auto c_ins    = r.instructions["c"];

        auto gemm = any_cast<rocblas_gemm<op::dot>>(gemm_ins->get_operator());

        // Already fused gemm
        if(not float_equal(gemm.beta, 0))
            return;
        gemm.beta = 1;

        if(not update_gemm(
               gemm, ins->module_inputs().front(), ins->inputs().front() == gemm_ins ? 0 : 1))
            return;

        // const-fold input if not standard shape since rocblas can't handle it
        if(not c_ins->get_shape().standard())
        {
            auto c = make_op("contiguous");
            auto l = c.compute(c.compute_shape({c_ins->get_shape()}), {c_ins->eval()});
            c_ins  = m.add_literal(l.get_shape(), l.data());
        }

        auto inputs = gemm_ins->inputs();
        inputs.pop_back();

        inputs.push_back(c_ins);
        inputs.push_back(ins->inputs().back());

        m.replace_instruction(ins, gemm, inputs);
    }
};

struct find_contiguous_tranpose_gemm
{
    auto matcher() const
    {
        return match::name("gpu::contiguous")(match::arg(0)(
            match::name("transpose")(
                match::arg(0)(match::name("gpu::gemm")(match::used_once()).bind("gemm")))
                .bind("transpose")));
    }

    template <class Vector>
    static bool is_swapped(const Vector& perm, std::size_t i, std::size_t j)
    {
        if(i >= perm.size() or j >= perm.size())
            return false;
        auto perm2 = perm;
        std::iota(perm2.begin(), perm2.end(), 0);
        std::swap(perm2[i], perm2[j]);
        return perm2 == perm;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto gemm      = r.instructions["gemm"];
        auto alloc     = gemm->inputs().back();
        auto transpose = r.instructions["transpose"];
        auto perm      = transpose->get_operator().to_value()["permutation"].to_vector<int64_t>();
        auto iperm     = invert_permutation(perm);

        if(perm.size() < 3)
            return;

        if(not is_swapped(perm, perm.size() - 3, perm.size() - 2))
            return;

        auto lens = gemm->get_shape().lens();
        if(lens.size() > 3 and
           not std::all_of(lens.begin(), lens.end() - 3, [](auto i) { return i == 1; }))
            return;

        auto gemmv           = gemm->get_operator().to_value();
        gemmv["trans_batch"] = 1;

        auto s = shape{alloc->get_shape().type(), reorder_dims(alloc->get_shape().lens(), iperm)};
        auto new_alloc = m.insert_instruction(gemm, make_op("allocate", {{"shape", to_value(s)}}));
        auto alloc_transpose =
            m.insert_instruction(gemm, make_op("transpose", {{"permutation", perm}}), new_alloc);

        auto inputs        = gemm->inputs();
        inputs.back()      = alloc_transpose;
        auto new_gemm      = m.insert_instruction(gemm, make_op("gpu::gemm", gemmv), inputs);
        auto gemm_transpoe = m.insert_instruction(gemm, transpose->get_operator(), new_gemm);

        m.replace_instruction(ins, gemm_transpoe);
    }
};

struct find_commutative_broadcast
{
    auto matcher() const
    {
        return match::name("gpu::add", "gpu::mul")(match::arg(1)(match::broadcast_shape()));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins  = r.result;
        auto args = ins->inputs();
        move_broadcasted_back(args);

        m.replace_instruction(ins, ins->get_operator(), args);
    }
};
} // namespace

struct find_contiguous
{
    auto matcher() const { return match::name("gpu::contiguous"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;

        m.replace_instruction(
            ins,
            make_op("gpu::precompile_op", {{"op", to_value(make_op("contiguous"))}}),
            ins->inputs());
    }
};

struct find_contiguous_pointwise
{
    auto matcher() const
    {
        return match::name("gpu::contiguous")(match::arg(0)(precompile_name("pointwise")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto pw     = ins->inputs().front();
        auto alloc  = ins->inputs().back();
        auto args   = pw->inputs();
        args.back() = alloc;

        m.replace_instruction(ins, pw->get_operator(), args, pw->module_inputs());
    }
};

void fuse_ops::apply(module& m) const
{
    // transformer中的gelu算子融合
    match::find_matches(m, find_contiguous_pointwise{}, find_gelu{}, find_gelu_new{fast_math});
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m, find_add_gelu{}, find_add_gelu_new{});

    // silu算子融合，yolov5/efficient
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m, find_sigmoid_mul{});
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m, find_add_sigmoid_mul{}, find_add_sigmoid_mul_add{});

    // 常见通用模式(注意：这些模式需要提前融合，后面的融合会用到这些融合)
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m,
                        find_triadd{},
                        find_mul_add{},
                        find_mul_add_relu{},
                        find_add_unary{"gpu::relu", hip_add_relu{}, hip_triadd_relu{}},
                        find_add_unary{"gpu::sigmoid", hip_add_sigmoid{}, hip_triadd_sigmoid{}},
                        find_add_unary{"gpu::tanh", hip_add_tanh{}, hip_triadd_tanh{}},
                        find_add_clip{});
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m,
                        find_mul_add_sqrt{},
                        find_mul_add_mul{},
                        find_mul_add_mul_tanh{},
                        find_mul_add_mul_tanh_add{},
                        find_div_mul_add{});

    // gemm融合
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m,
                        find_gemm_add{},
                        find_gemm_pointwise{},
                        find_contiguous_tranpose_gemm{},
                        find_commutative_broadcast{});

    // layernorm融合:add/triadd+layernorm
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m, find_layernorm{});
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m,
                        find_triadd_layernorm{},   // 动态模型中triadd+layernorm融合
                        find_add_layernorm{},      // 动态模型中add+layernorm融合
                        find_pointwise_layernorm{} // 静态模型中add/triadd+layernorm融合
    );

    // gpt2/bert算子融合
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(
        m, find_pow_mul_add_mul_tanh_add{}, find_pow_mul_add_mul_tanh_add_mul_mul{});

    // shape相关算子融合
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m, find_shape_convert{}, find_gather_for_shape{});
    run_passes(m, {dead_code_elimination{}});
    match::find_matches(m, find_shape_convert2{});

    // 如果是静态shape,则使用contiguous的静态实现
    if(!m.get_dynamic())
    {
        match::find_matches(m, find_contiguous{});
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
