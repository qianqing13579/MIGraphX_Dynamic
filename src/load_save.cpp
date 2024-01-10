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
#include <migraphx/load_save.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/json.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/ranges.hpp>
#include <fstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

program load(const std::string& filename, const file_options& options)
{
    return load_buffer(read_buffer(filename), options);
}
int load_mxr_version(const std::string& filename, const file_options& options)
{
    std::vector<char> buffer=read_buffer(filename);
    value v=from_msgpack(buffer.data(), buffer.size());
    int mxr_version = v.at("version").to<int>();
    return mxr_version;
}
program load_buffer(const std::vector<char>& buffer, const file_options& options)
{
    return load_buffer(buffer.data(), buffer.size(), options);
}
program load_buffer(const char* buffer, std::size_t size, const file_options& options)
{
    program p;
    
    // 加载mxr模型的时候,手动设置device_id
    p.set_device_id(options.device_id);

    // 加载模型
    if(options.format == "msgpack")
    {
        p.from_value(from_msgpack(buffer, size));
    }
    else if(options.format == "json")
    {
        p.from_value(from_json_string(buffer, size));
    }
    else
    {
        MIGRAPHX_THROW("Unknown format: " + options.format);
    }

    // 设置输入大小(在动态shape中需要判断是否超过最大值)
    for(auto ins : iterator_for(*p.get_main_module()))
    {
        if(ins->name() == "@param")
        {
            std::string param_name=any_cast<builtin::param>(ins->get_operator()).parameter;

            // 处理offload==false的情况
            if(!contains(param_name, "#output_"))
            {
                p.get_main_module()->set_input_shape(param_name,ins->get_shape());
            }
        }
    }
    return p;
}

void save(const program& p, const std::string& filename, const file_options& options)
{
    write_buffer(filename, save_buffer(p, options));
}
std::vector<char> save_buffer(const program& p, const file_options& options)
{
    value v = p.to_value();
    std::vector<char> buffer;
    if(options.format == "msgpack")
    {
        buffer = to_msgpack(v);
    }
    else if(options.format == "json")
    {
        std::string s = to_json_string(v);
        buffer        = std::vector<char>(s.begin(), s.end());
    }
    else
    {
        MIGRAPHX_THROW("Unknown format: " + options.format);
    }
    return buffer;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
