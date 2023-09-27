#ifndef MIGRAPHX_GUARD_RTGLIB_RESIZE_HPP
#define MIGRAPHX_GUARD_RTGLIB_RESIZE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/op/resize.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_resize
{
    op::resize op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::resize"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    // dynamic shape
    bool do_reshape(instruction_ref ins,std::unordered_map<instruction_ref, argument> &results)
    {
        // 不需要执行reshape,在运行期执行reshape
        return false;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
