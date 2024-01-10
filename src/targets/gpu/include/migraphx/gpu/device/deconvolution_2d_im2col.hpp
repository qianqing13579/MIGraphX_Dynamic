#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_DECONVOLUTION_2D_IM2COL_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_DECONVOLUTION_2D_IM2COL_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

/////////////////////// im2col ///////////////
shape compute_col_shape2(const shape& input,
                         const shape& weights,
                         const std::vector<std::size_t>& padding,
                         const std::vector<std::size_t>& stride,
                         const std::vector<std::size_t>& dilation);

void deconvolution_2d_im2col(context& ctx,
                             const argument& result,
                             const argument& x,
                             const argument& w,
                             const argument& col_buf,
                             const std::vector<std::size_t>& padding,
                             const std::vector<std::size_t>& stride,
                             const std::vector<std::size_t>& dilation,
                             int group,
                             bool is_1x1,
                             shape kernel_shape);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
