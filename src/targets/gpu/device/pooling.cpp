#include <migraphx/gpu/device/pooling.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <typename DataType>
__global__ void max_pool_kernel(int nthreads,
                                DataType* x,
                                int num,
                                int channels,
                                int height,
                                int width,
                                int pooled_height,
                                int pooled_width,
                                int kernel_h,
                                int kernel_w,
                                int stride_h,
                                int stride_w,
                                int pad_h,
                                int pad_w,
                                DataType* y)
{
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, nthreads)
    {
        int pw                 = index % pooled_width;
        int ph                 = (index / pooled_width) % pooled_height;
        int c                  = (index / pooled_width / pooled_height) % channels;
        int n                  = index / pooled_width / pooled_height / channels;
        int hstart             = ph * stride_h - pad_h;
        int wstart             = pw * stride_w - pad_w;
        int hend               = min(hstart + kernel_h, height);
        int wend               = min(wstart + kernel_w, width);
        hstart                 = max(hstart, 0);
        wstart                 = max(wstart, 0);
        DataType maxval        = -FLT_MAX;
        int maxidx             = -1;
        DataType* bottom_slice = x + (n * channels + c) * height * width;
        for(int h = hstart; h < hend; ++h)
        {
            for(int w = wstart; w < wend; ++w)
            {
                if(bottom_slice[h * width + w] > maxval)
                {
                    maxidx = h * width + w;
                    maxval = bottom_slice[maxidx];
                }
            }
        }
        y[index] = maxval;
    }
}

template <typename DataType>
__global__ void average_pool_kernel(int nthreads,
                                    DataType* x,
                                    int num,
                                    int channels,
                                    int height,
                                    int width,
                                    int pooled_height,
                                    int pooled_width,
                                    int kernel_h,
                                    int kernel_w,
                                    int stride_h,
                                    int stride_w,
                                    int pad_h,
                                    int pad_w,
                                    DataType* y)
{
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, nthreads)
    {
        int pw                 = index % pooled_width;
        int ph                 = (index / pooled_width) % pooled_height;
        int c                  = (index / pooled_width / pooled_height) % channels;
        int n                  = index / pooled_width / pooled_height / channels;
        int hstart             = ph * stride_h - pad_h;
        int wstart             = pw * stride_w - pad_w;
        int hend               = min(hstart + kernel_h, height + pad_h);
        int wend               = min(wstart + kernel_w, width + pad_w);
        int pool_size          = (hend - hstart) * (wend - wstart);
        hstart                 = max(hstart, 0);
        wstart                 = max(wstart, 0);
        hend                   = min(hend, height);
        wend                   = min(wend, width);
        DataType aveval        = 0;
        DataType* bottom_slice = x + (n * channels + c) * height * width;
        for(int h = hstart; h < hend; ++h)
        {
            for(int w = wstart; w < wend; ++w)
            {
                aveval += bottom_slice[h * width + w];
            }
        }
        y[index] = aveval / pool_size;
    }
}

void pooling(hipStream_t stream,
             const argument& result,
             const argument& x,
             const op::pooling_mode mode,
             const std::vector<std::size_t>& padding,
             const std::vector<std::size_t>& stride,
             const std::vector<std::size_t>& lengths,
             bool ceil_mode)
{
    int num = result.get_shape().elements();

    if(x.get_shape().type() == shape::float_type)
    {
        if(mode == op::pooling_mode::max)
        {
            max_pool_kernel<float><<<get_number_blocks(num), NUM_THREADS_PER_BLOCK, 0, stream>>>(
                num,
                (float*)x.data(),
                x.get_shape().lens()[0],
                x.get_shape().lens()[1],
                x.get_shape().lens()[2],
                x.get_shape().lens()[3],
                result.get_shape().lens()[2],
                result.get_shape().lens()[3],
                lengths[0],
                lengths[1],
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                (float*)result.data());
        }
        else if(mode == op::pooling_mode::average)
        {
            average_pool_kernel<float>
                <<<get_number_blocks(num), NUM_THREADS_PER_BLOCK, 0, stream>>>(
                    num,
                    (float*)x.data(),
                    x.get_shape().lens()[0],
                    x.get_shape().lens()[1],
                    x.get_shape().lens()[2],
                    x.get_shape().lens()[3],
                    result.get_shape().lens()[2],
                    result.get_shape().lens()[3],
                    lengths[0],
                    lengths[1],
                    stride[0],
                    stride[1],
                    padding[0],
                    padding[1],
                    (float*)result.data());
        }
    }
    else if(x.get_shape().type() == shape::half_type)
    {
        if(mode == op::pooling_mode::max)
        {
            max_pool_kernel<__fp16><<<get_number_blocks(num), NUM_THREADS_PER_BLOCK, 0, stream>>>(
                num,
                (__fp16*)x.data(),
                x.get_shape().lens()[0],
                x.get_shape().lens()[1],
                x.get_shape().lens()[2],
                x.get_shape().lens()[3],
                result.get_shape().lens()[2],
                result.get_shape().lens()[3],
                lengths[0],
                lengths[1],
                stride[0],
                stride[1],
                padding[0],
                padding[1],
                (__fp16*)result.data());
        }
        else if(mode == op::pooling_mode::average)
        {
            average_pool_kernel<__fp16>
                <<<get_number_blocks(num), NUM_THREADS_PER_BLOCK, 0, stream>>>(
                    num,
                    (__fp16*)x.data(),
                    x.get_shape().lens()[0],
                    x.get_shape().lens()[1],
                    x.get_shape().lens()[2],
                    x.get_shape().lens()[3],
                    result.get_shape().lens()[2],
                    result.get_shape().lens()[3],
                    lengths[0],
                    lengths[1],
                    stride[0],
                    stride[1],
                    padding[0],
                    padding[1],
                    (__fp16*)result.data());
        }
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
