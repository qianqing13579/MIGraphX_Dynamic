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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_pad_dynamic_edge : verify_program<test_pad_dynamic_edge>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {3, 3, 224, 224}};
        migraphx::shape s1{migraphx::shape::int64_type, {8}};
        std::vector<int64_t> pads1{0,0,1,1,0,0,1,1};
        std::vector<int64_t> pads2{1,1,1,1,1,1,1,1};

        auto l0  = mm->add_parameter("x", s);
        auto l1 = mm->add_literal(migraphx::literal{s1, pads1});
        auto l2 = mm->add_literal(migraphx::literal{s1, pads2});
        mm->add_instruction(migraphx::make_op("pad_dynamic", {{"mode", 2}, {"max_pads", pads1}}),l0,l1);
        mm->add_instruction(migraphx::make_op("pad_dynamic", {{"mode", 2}, {"max_pads", pads2}}),l0,l2);

        return p;
    }
};