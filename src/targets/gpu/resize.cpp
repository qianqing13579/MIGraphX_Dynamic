#include <migraphx/gpu/resize.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/upsample.hpp>
#include <migraphx/SimpleLog.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_resize::compute_shape(const std::vector<shape>& inputs) const
{
    return op.compute_shape({inputs.at(0)});
}

argument hip_resize::compute(context& ctx,
                             const shape& output_shape,
                             const std::vector<argument>& args) const
{
    std::vector<float> scales                  = op.scales;
    int mode                                   = op.mode;
    std::string coordinate_transformation_mode = op.coordinate_transformation_mode;

    // 不包含size参数
    if(scales.size() > 0)
    {
        argument result = args[args.size() - 1].reshape(this->compute_shape({args[0].get_shape()}));

        device::upsample(
            ctx.get_stream().get(), result, args[0], scales, mode, coordinate_transformation_mode);

        return result;
    }
    // 包含size参数
    else
    {
        // 判断哪个参数是shape参数
        argument shape_arg;
        for(int i = 1; i <= args.size() - 1; ++i)
        {
            if(args[i].get_shape().lens()[0] == args[0].get_shape().lens().size())
            {
                shape_arg = args[i];
                break;
            }
        }

        // 计算size，shape tensor在FP16模式下还是float类型
        std::vector<float> shape_data(shape_arg.get_shape().elements());

        // 拷贝到gpu
        hipMemcpyAsync(shape_data.data(),
                       shape_arg.data(),
                       shape_arg.get_shape().bytes(),
                       hipMemcpyDeviceToHost,
                       ctx.get_stream().get());

        ctx.finish();

        // 重新计算输出shape
        std::vector<int64_t> shape_data2(shape_arg.get_shape().elements());
        for(int i = 0; i < shape_data.size(); ++i)
        {
            shape_data2[i] = static_cast<int64_t>(shape_data[i]);
        }

        argument result =
            args[args.size() - 1].reshape(shape{args[0].get_shape().type(), shape_data2});

        device::upsample(
            ctx.get_stream().get(), result, args[0], scales, mode, coordinate_transformation_mode);

        return result;
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
