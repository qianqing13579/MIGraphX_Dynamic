#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_CONVOLUTION_2D_FP32_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_CONVOLUTION_2D_FP32_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void convolution_2d_fp32(hipStream_t stream, 
                const argument& result, 
                const argument& x,
                const argument& w,
                const std::vector<std::size_t> padding,
                const std::vector<std::size_t> stride,
                const std::vector<std::size_t> dilation,
                const int group);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
