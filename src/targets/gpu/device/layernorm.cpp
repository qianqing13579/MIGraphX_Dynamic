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
#include <migraphx/gpu/device/layernorm.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/pow.hpp>
#include <migraphx/gpu/device/fast_div.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

#ifndef MIGRAPHX_WORKAROUND_NAVI_DPP_SYNC
#if __AMDGCN_WAVEFRONT_SIZE == 32
#define MIGRAPHX_WORKAROUND_NAVI_DPP_SYNC 1
#else
#define MIGRAPHX_WORKAROUND_NAVI_DPP_SYNC 0
#endif
#endif

template <class T>
struct vector_type
{
};

template <class T, index_int N>
struct vector_type<vec<T, N>>
{
    using type = T;
};

template <class T>
using vector_type_t = typename vector_type<T>::type;

template <class T>
struct vector_size : std::integral_constant<index_int, 1>
{
};

template <class T, index_int N>
struct vector_size<vec<T, N>> : std::integral_constant<index_int, N>
{
};

template <class T, class F>
__device__ auto vec_transform(T x, F f)
{
    return f(x);
}

template <class T, index_int N, class F>
__device__ auto vec_transform(vec<T, N> x, F f)
{
    vec<T, N> y = x;
    // cppcheck-suppress useStlAlgorithm
    for(index_int k = 0; k < N; k++)
        y[k] = f(x[k]);
    return y;
}

template <class T, class U, class Op>
__device__ auto vec_reduce(T x, U, Op)
{
    return x;
}

template <class T, index_int N, class U, class Op>
__device__ auto vec_reduce(vec<T, N> x, U init, Op op)
{
    T r = init;
    for(index_int k = 0; k < N; k++)
        r = op(r, x[k]);
    return r;
}

template <index_int N, class Op, class T, class F>
__device__ auto auto_block_reduce(index idx, Op op, T init, index_int n, F f)
{
    auto r = block_reduce<N>(idx, op, init, n, f);
    return vec_reduce(r, 0, op);
}

template <index_int MaxBlockSize, class Input, class Output>
__device__ void layernorm(index_int i,
                          index idx,
                          std::size_t block_size_div,
                          index_int relements,
                          Input input,
                          Output output)
{
    using value_type       = decltype(input(idx.local));
    const auto relements_v = relements / vector_size<value_type>{};
    const auto out_idx     = fast_div(i, block_size_div);
    const auto base_idx    = out_idx * relements_v;
    const auto input_idx   = base_idx + idx.local;
    const bool in_range    = idx.local < relements_v;

    auto mean = [&](auto z) {
        auto m = auto_block_reduce<MaxBlockSize>(
                     idx, sum{}, value_type(0), relements_v, [=](auto) { return z; }) /
                 value_type(relements);
#if MIGRAPHX_WORKAROUND_NAVI_DPP_SYNC
        __builtin_amdgcn_s_barrier();
#endif
        return m;
    };

    // m = x - mean(x)
    value_type x = in_range ? input(input_idx) : 0;
    value_type m = x - mean(x);

    // mean(m ^ 2) + 1e-12
    value_type r = mean(m * m) + value_type(1e-12);

    // m * rsqrt(mean(m ^ 2) + 1e-12)
    if(in_range)
        output(input_idx, m * vec_transform(r, &rsqrt));
}

// m = x - mean(x)
// m / sqrt(mean(m ^ 2) + 1e-12)

template <index_int N, class Input, class Output, class... Arguments>
void layernorm_vec_impl(hipStream_t stream,
                        index_int nelements,
                        index_int relements,
                        Input in,
                        Output out,
                        const argument& result,
                        const Arguments&... args)
{
    hip_vec_visit_all<N>(result, args...)([&](auto output, auto... inputs) {
        const auto relements_v           = relements / N;
        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements_v, max_block_size);
        const std::size_t block_size_div = encode_divisor(block_size);
        assert(relements_v <= block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            layernorm<max_block_size>(
                i,
                idx,
                block_size_div,
                relements,
                [&](auto input_idx) { return in(inputs.data()[input_idx]...); },
                [&](auto input_idx, auto x) {
                    out(x, output.data()[input_idx], inputs.data()[input_idx]...);
                });
        });
    });
}

template <class Input, class Output, class... Arguments>
void layernorm_impl(hipStream_t stream,
                    index_int nelements,
                    index_int relements,
                    Input in,
                    Output out,
                    const argument& result,
                    const Arguments&... args)
{
    hip_visit_all(result, args...)([&](auto output, auto... inputs) {
        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);
        const std::size_t block_size_div = encode_divisor(block_size);
        assert(relements <= block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            layernorm<max_block_size>(
                i,
                idx,
                block_size_div,
                relements,
                [&](auto input_idx) { return in(inputs.data()[input_idx]...); },
                [&](auto input_idx, auto x) {
                    out(x, output.data()[input_idx], inputs.data()[input_idx]...);
                });
        });
    });
}

template <class... Arguments>
auto layernorm_fusion2(hipStream_t stream,
                       const argument& result,
                       const argument& arg1,
                       const Arguments&... args)
{
    return [=](auto input, auto output) {
        auto relements    = arg1.get_shape().lens().back();
        auto nelements    = result.get_shape().elements() / relements;
        auto output_shape = result.get_shape();
        auto reduce_output_lens(output_shape.lens());
        reduce_output_lens.back() = 1;

        if((relements % 4) == 0)
            layernorm_vec_impl<4>(
                stream, nelements, relements, input, output, result, arg1, args...);
        else if(relements < 256)
            layernorm_impl(stream, nelements, relements, input, output, result, arg1, args...);
        else
            MIGRAPHX_THROW("No kernel for layernorm");
    };
}

// layernorm融合算子
template <class Input, class... Arguments>
void layernorm_fusion(hipStream_t stream,
                      std::size_t relements,
                      Input in,
                      const argument& result,
                      const Arguments&... args)
{
    auto nelements = result.get_shape().elements() / relements;

    hip_visit_all(result, args...)([&](auto output, auto... input) {
        using value_type = typename decltype(output)::value_type;

        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);
        const std::size_t block_size_div = encode_divisor(block_size);

        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx  = i / block_size;
            const auto base_idx = out_idx * relements;

            // 保存每个线程执行local_stride访问的元素，比如relements为4096，则每个线程需要local_stride
            // 16次，目前支持的最大relements为16*256
            float x_data[16];
            auto x = [&](auto j) -> float& {
                return x_data[fast_div(j, block_size_div)];
            }; // j / block_size

            idx.local_stride(relements,
                             [&](auto j) __device__ { x(j) = (float)in(input[base_idx + j]...); });

            // mean(x)
            float mean_x = block_reduce<max_block_size>(
                              idx, sum{}, 0, relements, [&](auto j) __device__ { return x(j); }) /
                          relements; // x(j)等价于input[base_idx + j];

            // m = x - mean(x)
            idx.local_stride(relements, [&](auto j) __device__ { x(j) = x(j) - mean_x; });

            // mean(m ^ 2)
            float pow_mean =
                block_reduce<max_block_size>(
                    idx, sum{}, 0, relements, [&](auto j) __device__ { return (float)x(j) * x(j); }) /
                relements;

            // m * rsqrt(mean(m ^ 2) + 1e-12)
            idx.local_stride(relements, [&](auto j) __device__ {
                output[base_idx + j] = (value_type)(x(j) * ::rsqrt(pow_mean + 1e-12));
            });
        });
    });
}

void layernorm(hipStream_t stream, const argument& result, const argument& arg1)
{
    layernorm_fusion(
        stream, arg1.get_shape().lens().back(), [](auto x) { return x; }, result, arg1);
}

void add_layernorm(hipStream_t stream,
                   const argument& result,
                   const argument& arg1,
                   const argument& arg2)
{
    layernorm_fusion(
        stream,
        arg1.get_shape().lens().back(),
        [](auto x, auto y) { return x + y; },
        result,
        arg1,
        arg2);
}

void triadd_layernorm(hipStream_t stream,
                      const argument& result,
                      const argument& arg1,
                      const argument& arg2,
                      const argument& arg3)
{
    layernorm_fusion(
        stream,
        arg1.get_shape().lens().back(),
        [](auto x, auto y, auto z) { return x + y + z; },
        result,
        arg1,
        arg2,
        arg3);
}

///////////////////////////////// 使用shuffle指令实现LayerNorm ///////////////////////////
template <int WarpSize, typename T>
__inline__ __device__ T warp_reduce_sum(T val)
{
#pragma unroll
    for(int offset = WarpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

template <int WarpSize, typename T>
__inline__ __device__ T block_reduce_sum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x % WarpSize;
    int wid  = threadIdx.x / WarpSize;

    val = warp_reduce_sum<WarpSize, T>(val);

    if(lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x / WarpSize)) ? shared[lane] : (T)(0.0f);
    if(wid == 0)
    {
        val = warp_reduce_sum<WarpSize, T>(val);
    }

    return val;
}

// 使用混合精度计算
template <int BlockSize, int WarpSize, typename T>
__global__ void layernorm_kernel(const T* input,
                                 T* output,
                                 const int size // 每个block负责size个数据的规约
)
{
    const int block_offset = blockIdx.x * size;
    const T* ptr           = input + block_offset;
    T* dst                 = output + block_offset;

    __shared__ float mean_x;
    __shared__ float pow_mean;

    // 保存每个线程需要stride的数据
    float x_data[16];
    auto x            = [&](auto j) -> float& { return x_data[j / BlockSize]; };
    float thread_sum1 = 0.f; // 每个线程需要local_stride的所有数据和
    for(int i = threadIdx.x; i < size; i += BlockSize)
    {
        float value = (float)(ptr[i]);
        x(i)        = value;
        thread_sum1 += value;
    }

    // mean(x)
    float mean = block_reduce_sum<WarpSize, float>(thread_sum1);
    if(threadIdx.x == 0)
    {
        mean_x = mean / size;
    }
    __syncthreads(); // 这里需要加同步

    // x-mean(x)
    float thread_sum2 = 0.f;
    for(int i = threadIdx.x; i < size; i += BlockSize)
    {
        x(i) -= mean_x;
        thread_sum2 += (x(i) * x(i));
    }

    // pow
    float pow_mean_x = block_reduce_sum<WarpSize, float>(thread_sum2);
    if(threadIdx.x == 0)
    {
        pow_mean = pow_mean_x / size;
    }
    __syncthreads(); // 这里需要加同步

    // x-mean(x)
    for(int i = threadIdx.x; i < size; i += BlockSize)
    {
        dst[i] = (T)(x(i) * ::rsqrt(pow_mean + 1e-12));
    }
}

template <int BlockSize, int WarpSize, typename T>
__global__ void layernorm_kernel2(const T* input,
                                  T* output,
                                  const int size // 每个block负责size个数据的规约
)
{
    __shared__ float ssum1[BlockSize / WarpSize];
    __shared__ float ssum2[BlockSize / WarpSize];
    __shared__ float mean;
    __shared__ float var;

    const int block_offset = blockIdx.x * size;
    const T* ptr           = input + block_offset;
    T* dst                 = output + block_offset;

    float thread_sum1 = 0.f;
    float thread_sum2 = 0.f;

    int lane = threadIdx.x % WarpSize;
    int wid  = threadIdx.x / WarpSize;

    float x_data[16];
    auto x = [&](auto j) -> float& { return x_data[j / BlockSize]; };
    for(int i = threadIdx.x; i < size; i += BlockSize)
    {
        float value = (float)(ptr[i]);
        x(i)        = value;
        thread_sum1 += value;
        thread_sum2 += value * value;
    }

    thread_sum1 = warp_reduce_sum<WarpSize, float>(thread_sum1);
    thread_sum2 = warp_reduce_sum<WarpSize, float>(thread_sum2);
    if(lane == 0)
    {
        ssum1[wid] = thread_sum1;
        ssum2[wid] = thread_sum2;
    }
    __syncthreads();

    if(threadIdx.x < blockDim.x / WarpSize)
    {
        thread_sum1 = ssum1[threadIdx.x];
        thread_sum2 = ssum2[threadIdx.x];
    }
    else
    {
        thread_sum1 = 0;
        thread_sum2 = 0;
    }

    // 计算第一个warp
    if(wid == 0)
    {
        thread_sum1 = warp_reduce_sum<WarpSize, float>(thread_sum1);
        thread_sum2 = warp_reduce_sum<WarpSize, float>(thread_sum2);
    }

    if(threadIdx.x == 0)
    {
        mean = thread_sum1 / size;
        var  = (thread_sum2 / size - mean * mean);
        var  = 1.0 / sqrt(var + 1e-12);
    }
    __syncthreads();

    float b = -mean * var;
    for(int i = threadIdx.x; i < size; i += BlockSize)
    {
        dst[i] = (T)(x(i) * var + b);
    }
}

// void layernorm(hipStream_t stream, const argument& result, const argument& arg1)
// {
//     int n             = result.get_shape().elements() / arg1.get_shape().lens().back();
//     int reduce_number = arg1.get_shape().lens().back(); // 每个block需要规约的元素个数

//     const int block_size = 256;
//     const int warp_size  = 64;
//     if(arg1.get_shape().type() == shape::float_type)
//     {
//         layernorm_kernel<block_size, warp_size, float><<<n, block_size, 0, stream>>>(
//             (float*)arg1.data(), (float*)result.data(), reduce_number);
//         // layernorm_kernel2<block_size, warp_size, float, float><<<n, block_size, 0, stream>>>(
//         //     (float*)arg1.data(), (float*)result.data(), reduce_number);
//     }
//     else if(arg1.get_shape().type() == shape::half_type)
//     {
//         layernorm_kernel<block_size, warp_size, __fp16><<<n, block_size, 0, stream>>>(
//             (__fp16*)arg1.data(), (__fp16*)result.data(), reduce_number);
//         // layernorm_kernel2<block_size, warp_size, __fp16, float><<<n, block_size, 0, stream>>>(
//         //     (__fp16*)arg1.data(), (__fp16*)result.data(), reduce_number);
//     }
// }

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
