/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

const auto& get_nearest_op(const std::string& mode)
{
    using nearest_op = std::function<std::size_t(std::size_t, double)>;
    static std::unordered_map<std::string, nearest_op> const nearest_ops = {
        {"round_prefer_floor",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::ceil((val - 0.5)));
         }},
        {"round_prefer_ceil",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::round((val)));
         }},
        {"floor",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::floor((val)));
         }},
        {"ceil", [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::ceil((val)));
         }}};

    if(not contains(nearest_ops, mode))
    {
        MIGRAPHX_THROW("PARSE_RESIZE: nearest_mode " + mode + " not supported!");
    }

    return nearest_ops.at(mode);
}

const auto& get_original_idx_op(const std::string& mode)
{
    using original_idx_op = std::function<double(std::size_t, std::size_t, std::size_t, double)>;
    static std::unordered_map<std::string, original_idx_op> const idx_ops = {
        {"half_pixel",
         [=](std::size_t, std::size_t, std::size_t idx, double scale) {
             return (idx + 0.5) / scale - 0.5;
         }},
        {"pytorch_half_pixel",
         [=](std::size_t, std::size_t l_out, std::size_t idx, double scale) {
             return l_out > 1 ? (idx + 0.5) / scale - 0.5 : 0.0;
         }},
        {"align_corners",
         [=](std::size_t l_in, std::size_t l_out, std::size_t idx, double) {
             return (l_out == 1) ? 0.0 : (1.0 * idx * (l_in - 1.0) / (l_out - 1.0));
         }},
        {"asymmetric",
         [=](std::size_t, std::size_t, std::size_t idx, double scale) { return idx / scale; }},
        {"tf_half_pixel_for_nn", [=](std::size_t, std::size_t, std::size_t idx, double scale) {
             return (idx + 0.5) / scale;
         }}};

    if(not contains(idx_ops, mode))
    {
        MIGRAPHX_THROW("PARSE_RESIZE: coordinate_transformation_mode " + mode + " not supported!");
    }

    return idx_ops.at(mode);
}

static std::vector<int>
calc_neighbor_points(const std::vector<std::vector<std::vector<std::size_t>>>& vvv_ind,
                     int i_dim,
                     const std::vector<std::vector<std::size_t>>& vec_dims,
                     const shape& in_s)
{
    if(i_dim == vvv_ind.size())
    {
        std::vector<int> vec_ind;
        vec_ind.resize(vec_dims.size());
        std::transform(vec_dims.begin(), vec_dims.end(), vec_ind.begin(), [&](auto idx) {
            return static_cast<int>(in_s.index(idx));
        });

        return vec_ind;
    }

    const auto& vv_ind = vvv_ind[i_dim];
    const auto& vv_lo  = vv_ind.at(0);
    std::vector<std::vector<std::size_t>> vec_dims1;
    for(std::size_t start = 0; start < vec_dims.size(); start += vv_lo.size())
    {
        std::transform(vv_lo.begin(),
                       vv_lo.end(),
                       vec_dims.begin() + start,
                       std::back_inserter(vec_dims1),
                       [](auto i, auto dim) {
                           dim.push_back(i);
                           return dim;
                       });
    }

    const auto& vv_hi = vv_ind.at(1);
    for(std::size_t start = 0; start < vec_dims.size(); start += vv_lo.size())
    {
        std::transform(vv_hi.begin(),
                       vv_hi.end(),
                       vec_dims.begin() + start,
                       std::back_inserter(vec_dims1),
                       [](auto i, auto dim) {
                           dim.push_back(i);
                           return dim;
                       });
    }

    return calc_neighbor_points(vvv_ind, i_dim + 1, vec_dims1, in_s);
}

static std::string get_coord_trans_mode(const onnx_parser::attribute_map& attr)
{
    std::string coord_trans_mode = "half_pixel";
    if(contains(attr, "coordinate_transformation_mode"))
    {
        coord_trans_mode = attr.at("coordinate_transformation_mode").s();
        // does not support transformation mode "tf_crop_and_resize"
        if(coord_trans_mode == "tf_crop_and_resize")
        {
            MIGRAPHX_THROW("PARSE_RESIZE: \"tf_crop_and_resize\" mode is not supported!");
        }
    }

    return coord_trans_mode;
}

static std::string get_mode(const onnx_parser::attribute_map& attr)
{
    std::string mode = "nearest";
    if(contains(attr, "mode"))
    {
        mode = attr.at("mode").s();
        if(mode != "nearest" and mode != "linear")
        {
            MIGRAPHX_THROW("PARSE_RESIZE: only nearest and linear modes are supported!");
        }
    }

    return mode;
}

static std::string get_nearest_mode(const onnx_parser::attribute_map& attr)
{
    std::string nearest_mode = "round_prefer_floor";
    if(contains(attr, "nearest_mode"))
    {
        nearest_mode = attr.at("nearest_mode").s();
    }

    return nearest_mode;
}

struct parse_resize : op_parser<parse_resize>
{
    std::vector<op_desc> operators() const { return {{"Resize"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        // coord transform mode
        std::string coord_trans_mode = get_coord_trans_mode(info.attributes);

        // opset-10中resize的coord_trans_mode为asymmetric
        if(parser.model_opset_version <= 10)
        {
            coord_trans_mode = "asymmetric";
        }

        // mode: only nearest and linear modes are supported for now
        std::string mode = get_mode(info.attributes);

        // nearest mode
        std::string nearest_mode = get_nearest_mode(info.attributes);

        // check exclude_outside, only support 0
        if(contains(info.attributes, "exclude_outside") and
           info.attributes.at("exclude_outside").i() == 1)
        {
            MIGRAPHX_THROW("PARSE_" + opd.op_name + ": exclude_outside 1 is not supported!");
        }

        // input data shape info
        auto in_s    = args[0]->get_shape();
        auto in_lens = in_s.lens();

        std::vector<std::size_t> out_lens(in_lens.size()); // sizes
        std::vector<double> vec_scale;                     // scale
        std::vector<instruction_ref> new_args;
        new_args.push_back(args[0]);

        // 遍历输入参数，计算scale和out_lens
        for(int i = 1; i < args.size(); ++i)
        {
            instruction_ref arg = args[i];
            if(arg->name() == "undefined")
            {
                continue;
            }

            // skipped empty input
            auto lens = arg->get_shape().lens();
            if(lens.empty())
            {
                continue;
            }

            // undefined和空的参数都去掉
            new_args.push_back(arg);

            auto type = arg->get_shape().type();

            // 这是sizes参数，表示输出大小
            // output size
            if(type == shape::int64_type)
            {
                auto arg_out_s = arg->eval_for_shape();
                check_arg_empty(arg_out_s,
                                "PARSE_" + opd.op_name + ": dynamic output size is not supported!");
                arg_out_s.visit([&](auto ol) { out_lens.assign(ol.begin(), ol.end()); });

                if(out_lens.size() != in_lens.size())
                {
                    MIGRAPHX_THROW("PARSE_" + opd.op_name +
                                   ": specified output size does not match input size");
                }
            }
            // scales参数
            else if(lens[0] == in_lens.size())
            {
                auto arg_scale = arg->eval_for_shape();
                check_arg_empty(arg_scale,
                                "PARSE_" + opd.op_name + ": dynamic input scale is not supported!");

                arg_scale.visit([&](auto v) { vec_scale.assign(v.begin(), v.end()); });
                if(in_lens.size() != vec_scale.size())
                {
                    MIGRAPHX_THROW("PARSE_" + opd.op_name +
                                   ": ranks of input and scale are different!");
                }

                std::transform(
                    in_lens.begin(),
                    in_lens.end(),
                    vec_scale.begin(),
                    out_lens.begin(),
                    [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });
            }
        }

        int mode_ = 0;
        if(mode == "nearest")
        {
            mode_ = 1;
        }
        else
        {
            mode_ = 2;
        }
        return info.add_instruction(make_op("resize",
                                            {{"scales", vec_scale},
                                             {"max_size", out_lens},
                                             {"mode", mode_},
                                             {"coordinate_transformation_mode", coord_trans_mode}}),
                                    new_args);
    }
};

} // namespace onnx

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
