#ifndef MIGRAPHX_GUARD_OPERATORS_RESIZE_HPP
#define MIGRAPHX_GUARD_OPERATORS_RESIZE_HPP

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

struct resize
{
    std::vector<float> scales;
    std::vector<int64_t> sizes;
    int mode = 0; // 1: nereast 2: bilinear/linear 3: cubic
    std::string coordinate_transformation_mode;

    std::string name() const { return "resize"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.scales, "scales"),
                    f(self.sizes, "max_size"),
                    f(self.mode, "mode"),
                    f(self.coordinate_transformation_mode, "coordinate_transformation_mode"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{{inputs[0]}, *this}.has(1);

        if(!scales.empty())
        {
            // 计算输出blob大小
            auto in_s    = inputs[0];
            auto in_lens = in_s.lens();
            if(in_lens.size() != scales.size())
            {
                MIGRAPHX_THROW("PARSE_UPSAMPLE: ranks of input and scale are different!");
            }
            std::vector<std::size_t> out_lens(in_lens.size());
            std::transform(
                in_lens.begin(),
                in_lens.end(),
                scales.begin(),
                out_lens.begin(),
                [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });

            return shape{in_s.type(), out_lens};
        }
        else if(!sizes.empty())
        {
            return shape{inputs[0].type(), sizes};
        }
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
