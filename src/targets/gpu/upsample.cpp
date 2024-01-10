#include <migraphx/gpu/upsample.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/upsample.hpp>
#include <migraphx/SimpleLog.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_upsample::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2).standard(); // 注意，这里有两个参数，输入和输出
    return op.compute_shape({inputs.at(0)});
}

argument hip_upsample::compute(context& ctx,
                               const shape& output_shape,
                               const std::vector<argument>& args) const
{
    std::vector<float> scales = op.scales;
    int mode                  = op.mode;

    device::upsample(ctx.get_stream().get(), args[1], args[0], scales, mode, "asymmetric");

    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
