#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_POOLING_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_POOLING_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void pooling(hipStream_t stream, 
            const argument& result, 
            const argument& x,
            const op::pooling_mode mode,
            const std::vector<std::size_t>& padding,
            const std::vector<std::size_t>& stride,
            const std::vector<std::size_t>& lengths,
            bool ceil_mode=false);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
