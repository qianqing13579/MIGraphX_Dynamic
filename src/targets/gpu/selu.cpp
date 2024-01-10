#include <migraphx/gpu/selu.hpp>
#include <migraphx/gpu/device/selu.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_selu::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard(); // 注意，这里有两个参数，输入和输出
    return op.compute_shape({inputs.at(0)});
}

argument
hip_selu::compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
{

    device::selu(ctx.get_stream().get(), args[1], args[0], op.alpha, op.gamma);

    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
