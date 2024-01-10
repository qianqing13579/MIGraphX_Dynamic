#ifndef MIGRAPHX_GUARD_OPERATORS_UPSAMPLE_HPP
#define MIGRAPHX_GUARD_OPERATORS_UPSAMPLE_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct upsample
{
    std::vector<float> scales;
    int mode          = 0; // 1: nereast 2: bilinear/linear 3: cubic
    int align_corners = 0;

    std::string name() const { return "upsample"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.scales, "scales"), f(self.mode, "mode"), f(self.align_corners, "align_corners"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();

        // 计算输出blob大小
        auto in_s    = inputs[0];
        auto in_lens = in_s.lens();
        if(in_lens.size() != scales.size())
        {
            MIGRAPHX_THROW("PARSE_UPSAMPLE: ranks of input and scale are different!");
        }
        std::vector<std::size_t> out_lens(in_lens.size());
        std::transform(in_lens.begin(),
                       in_lens.end(),
                       scales.begin(),
                       out_lens.begin(),
                       [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });

        return shape{in_s.type(), out_lens};
        ;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
