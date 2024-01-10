#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_UPSAMPLE_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_UPSAMPLE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void upsample(hipStream_t stream,
              const argument& result,
              const argument& arg,
              const std::vector<float>& scales,
              int mode,
              std::string coordinate_transformation_mode);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
