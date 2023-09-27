#ifndef MIGRAPHX_GUARD_RTGLIB_UPSAMPLE_HPP
#define MIGRAPHX_GUARD_RTGLIB_UPSAMPLE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/op/upsample.hpp>
#include <migraphx/reflect.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_upsample
{
    op::upsample op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::upsample"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
