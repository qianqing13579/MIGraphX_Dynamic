#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_upsample : op_parser<parse_upsample>
{
    std::vector<op_desc> operators() const { return {{"Upsample"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        if(contains(info.attributes, "mode"))
        {
            auto mode = info.attributes.at("mode").s();
            if(mode != "nearest")
            {
                MIGRAPHX_THROW("PARSE_UPSAMPLE: only nearest mode is supported!");
            }
        }

        auto arg_scale = args[1]->eval_for_shape();
        check_arg_empty(arg_scale, "PARSE_UPSAMPLE: only constant scale is supported!");
        std::vector<float> scales;
        arg_scale.visit([&](auto v) { scales.assign(v.begin(), v.end()); });

        return info.add_instruction(
            make_op("upsample", {{"scales", scales}, {"mode", 1}, {"align_corners", 0}}),
            {args[0]});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
