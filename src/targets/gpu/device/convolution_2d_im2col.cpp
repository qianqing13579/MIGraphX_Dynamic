#include <migraphx/gpu/device/convolution_2d_im2col.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/gemm_impl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

//////////////////////////// im2col ///////////////////////////////
shape compute_col_shape(const shape& input,
                        const shape& weights,
                        const std::vector<std::size_t>& padding,
                        const std::vector<std::size_t>& stride,
                        const std::vector<std::size_t>& dilation)
{
    // 先计算col图像的shape
    auto input_channels = input.lens()[1];
    auto kernel_height  = weights.lens()[2];
    auto kernel_width   = weights.lens()[3];

    auto padding_h = 2 * padding[0];
    auto padding_w = 2 * padding[1];
    auto output_height =
        (input.lens()[2] - (1 + dilation[0] * (kernel_height - 1)) + padding_h) / stride[0] + 1;
    auto output_width =
        (input.lens()[3] - (1 + dilation[1] * (kernel_width - 1)) + padding_w) / stride[1] + 1;

    auto channels_col = kernel_height * kernel_width * input_channels;

    // col图像的行：channels_col，列：output_height * output_width
    return {input.type(), {channels_col, output_height * output_width}};
}

template <typename DataType>
__global__ void im2col_gpu_kernel(int n,
                                  const DataType* data_im,
                                  int height,
                                  int width,
                                  int kernel_h,
                                  int kernel_w,
                                  int pad_h,
                                  int pad_w,
                                  int stride_h,
                                  int stride_w,
                                  int dilation_h,
                                  int dilation_w,
                                  int height_col,
                                  int width_col,
                                  DataType* data_col)
{
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, n)
    {
        // 每个线程的多维索引
        int w_col   = index % width_col; // w
        int h_index = index / width_col;
        int h_col   = h_index % height_col; // h
        int c_im    = h_index / height_col; // c

        // 每个线程对应的滑动窗口在原图中的索引
        int h_offset =
            h_col * stride_h - pad_h; // 注意，由于有padding的存在，所以索引需要减去padding
        int w_offset = w_col * stride_w - pad_w;

        // 该线程对应的输入tensor的数据指针
        const DataType* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset; // 要拷贝的原图像起始位置

        // 该线程对应的输出tensor的数据指针
        int c_col              = c_im * kernel_h * kernel_w;
        DataType* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        for(int i = 0; i < kernel_h; ++i)
        {
            for(int j = 0; j < kernel_w; ++j)
            {
                int h_im      = h_offset + i * dilation_h;
                int w_im      = w_offset + j * dilation_w;
                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                                    ? data_im_ptr[i * dilation_h * width + j * dilation_w]
                                    : 0;
                data_col_ptr += height_col *
                                width_col; // 步长为height_col * width_col，即每个滑动窗口展开为一列
            }
        }
    }
}

template <typename DataType>
__global__ void im2col_gpu_3x3_kernel(int n,
                                      const DataType* data_im,
                                      int height,
                                      int width,
                                      int kernel_h,
                                      int kernel_w,
                                      int pad_h,
                                      int pad_w,
                                      int stride_h,
                                      int stride_w,
                                      int dilation_h,
                                      int dilation_w,
                                      int height_col,
                                      int width_col,
                                      DataType* data_col)
{
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, n)
    {
        // 每个线程的多维索引
        int w_col   = index % width_col; // w
        int h_index = index / width_col;
        int h_col   = h_index % height_col; // h
        int c_im    = h_index / height_col; // c

        // 每个线程对应的滑动窗口在原图中的索引
        int h_offset =
            h_col * stride_h - pad_h; // 注意，由于有padding的存在，所以索引需要减去padding
        int w_offset = w_col * stride_w - pad_w;

        // 该线程对应的输入tensor的数据指针
        const DataType* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset; // 要拷贝的原图像起始位置

        // 该线程对应的输出tensor的数据指针
        int c_col              = c_im * kernel_h * kernel_w;
        DataType* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
#pragma unroll
        for(int i = 0; i < 3; ++i)
        {
#pragma unroll
            for(int j = 0; j < 3; ++j)
            {
                int h_im      = h_offset + i;
                int w_im      = w_offset + j;
                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                                    ? data_im_ptr[i * width + j]
                                    : 0;
                data_col_ptr += height_col *
                                width_col; // 步长为height_col * width_col，即每个滑动窗口展开为一列
            }
        }
    }
}

template <typename DataType>
void im2col_gpu(hipStream_t stream,
                const DataType* data_im,
                int channels,
                int height,
                int width,
                int kernel_h,
                int kernel_w,
                int pad_h,
                int pad_w,
                int stride_h,
                int stride_w,
                int dilation_h,
                int dilation_w,
                DataType* data_col)
{
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col  = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // 启动 channels * height_col * width_col 个线程
    int num_kernels              = channels * height_col * width_col;
    int number_threads_per_block = 256;
    dim3 threads(number_threads_per_block);
    dim3 blocks((num_kernels + number_threads_per_block - 1) / number_threads_per_block);

    if(kernel_h == 3 && kernel_w == 3 && dilation_h == 1 && dilation_w == 1)
    {
        // 循环展开
        im2col_gpu_3x3_kernel<DataType><<<blocks, threads, 0, stream>>>(num_kernels,
                                                                        data_im,
                                                                        height,
                                                                        width,
                                                                        kernel_h,
                                                                        kernel_w,
                                                                        pad_h,
                                                                        pad_w,
                                                                        stride_h,
                                                                        stride_w,
                                                                        dilation_h,
                                                                        dilation_w,
                                                                        height_col,
                                                                        width_col,
                                                                        data_col);
    }
    else
    {
        im2col_gpu_kernel<DataType><<<blocks, threads, 0, stream>>>(num_kernels,
                                                                    data_im,
                                                                    height,
                                                                    width,
                                                                    kernel_h,
                                                                    kernel_w,
                                                                    pad_h,
                                                                    pad_w,
                                                                    stride_h,
                                                                    stride_w,
                                                                    dilation_h,
                                                                    dilation_w,
                                                                    height_col,
                                                                    width_col,
                                                                    data_col);
    }
}

void convolution_2d_im2col(context& ctx,
                           const argument& result,
                           const argument& x,
                           const argument& w,
                           const argument& col_buf,
                           const std::vector<std::size_t>& padding,
                           const std::vector<std::size_t>& stride,
                           const std::vector<std::size_t>& dilation,
                           int group,
                           bool is_1x1)
{
    // 对col图像进行reshape
    argument col_buffer;
    shape new_col_shape =
        compute_col_shape(x.get_shape(), w.get_shape(), padding, stride, dilation);
    if(!is_1x1)
    {
        col_buffer = col_buf.reshape(new_col_shape);
    }

    int batch_size = x.get_shape().lens()[0];
    int number_of_one_batch_input =
        x.get_shape().lens()[1] * x.get_shape().lens()[2] * x.get_shape().lens()[3];
    int number_of_one_batch_output =
        result.get_shape().lens()[1] * result.get_shape().lens()[2] * result.get_shape().lens()[3];

    std::size_t w0   = w.get_shape().lens()[0];
    std::size_t w123 = w.get_shape().lens()[1] * w.get_shape().lens()[2] * w.get_shape().lens()[3];
    std::size_t output_image_size = new_col_shape.lens()[1];
    if(x.get_shape().type() == shape::float_type)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // 每次只处理batch=1的数据,防止内存溢出
            if(!is_1x1)
            {
                im2col_gpu(ctx.get_stream().get(),
                           (float*)x.data() + i * number_of_one_batch_input,
                           x.get_shape().lens()[1],
                           x.get_shape().lens()[2],
                           x.get_shape().lens()[3],
                           w.get_shape().lens()[2],
                           w.get_shape().lens()[3],
                           padding[0],
                           padding[1],
                           stride[0],
                           stride[1],
                           dilation[0],
                           dilation[1],
                           (float*)col_buffer.data());
            }
            else
            {
                col_buffer =
                    argument{new_col_shape, (float*)x.data() + i * number_of_one_batch_input};
            }

            float alpha = 1.0;
            float beta  = 0.0;

            // 计算一个batch的卷积
            argument A = w.reshape(shape{w.get_shape().type(), {w0, w123}});
            argument B = col_buffer;
            argument C = argument{
                shape{A.get_shape().type(), {A.get_shape().lens()[0], B.get_shape().lens()[1]}},
                (float*)result.data() + i * number_of_one_batch_output};
            if(group == 1)
            {
                gemm(ctx, C.get_shape(), {A, B, C}, alpha, beta, true, false);
            }
            else
            {
                argument A1 = w.reshape(shape{w.get_shape().type(), {w0, 1, w123}});
                argument B1 = col_buffer.reshape(
                    shape{col_buffer.get_shape().type(), {w0, w123, output_image_size}});
                argument C1 = argument{shape{A.get_shape().type(), {w0, 1, output_image_size}},
                                       (float*)result.data() + i * number_of_one_batch_output};
                gemm(ctx, C1.get_shape(), {A1, B1, C1}, alpha, beta, true, false);
            }
        }
    }
    else if(x.get_shape().type() == shape::half_type)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // 每次只处理batch=1的数据,防止内存溢出
            if(!is_1x1)
            {
                im2col_gpu(ctx.get_stream().get(),
                           (__fp16*)x.data() + i * number_of_one_batch_input,
                           x.get_shape().lens()[1],
                           x.get_shape().lens()[2],
                           x.get_shape().lens()[3],
                           w.get_shape().lens()[2],
                           w.get_shape().lens()[3],
                           padding[0],
                           padding[1],
                           stride[0],
                           stride[1],
                           dilation[0],
                           dilation[1],
                           (__fp16*)col_buffer.data());
            }
            else
            {
                col_buffer =
                    argument{new_col_shape, (__fp16*)x.data() + i * number_of_one_batch_input};
            }

            float alpha = 1.0;
            float beta  = 0.0;

            // 计算一个batch的卷积
            argument A = w.reshape(shape{w.get_shape().type(), {w0, w123}});
            argument B = col_buffer;
            argument C = argument{
                shape{A.get_shape().type(), {A.get_shape().lens()[0], B.get_shape().lens()[1]}},
                (__fp16*)result.data() + i * number_of_one_batch_output};
            if(group == 1)
            {
                gemm(ctx, C.get_shape(), {A, B, C}, alpha, beta, true, false);
            }
            else
            {
                argument A1 = w.reshape(shape{w.get_shape().type(), {w0, 1, w123}});
                argument B1 = col_buffer.reshape(
                    shape{col_buffer.get_shape().type(), {w0, w123, output_image_size}});
                argument C1 = argument{shape{A.get_shape().type(), {w0, 1, output_image_size}},
                                       (__fp16*)result.data() + i * number_of_one_batch_output};
                gemm(ctx, C1.get_shape(), {A1, B1, C1}, alpha, beta, true, false);
            }
        }
    }
    else if(x.get_shape().type() == shape::int8_type)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // 每次只处理batch=1的数据,防止内存溢出
            if(!is_1x1)
            {
                im2col_gpu(ctx.get_stream().get(),
                           (int8_t*)x.data() + i * number_of_one_batch_input,
                           x.get_shape().lens()[1],
                           x.get_shape().lens()[2],
                           x.get_shape().lens()[3],
                           w.get_shape().lens()[2],
                           w.get_shape().lens()[3],
                           padding[0],
                           padding[1],
                           stride[0],
                           stride[1],
                           dilation[0],
                           dilation[1],
                           (int8_t*)col_buffer.data());
            }
            else
            {
                col_buffer =
                    argument{new_col_shape, (int8_t*)x.data() + i * number_of_one_batch_input};
            }

            float alpha = 1.0;
            float beta  = 0.0;

            // 计算一个batch的卷积
            argument A = w.reshape(shape{w.get_shape().type(), {w0, w123}});
            argument B = col_buffer;
            argument C = argument{
                shape{A.get_shape().type(), {A.get_shape().lens()[0], B.get_shape().lens()[1]}},
                (int8_t*)result.data() + i * number_of_one_batch_output};
            if(group == 1)
            {
                gemm(ctx, C.get_shape(), {A, B, C}, int32_t(alpha), int32_t(beta), false, false);
            }
            else
            {
                argument A1 = w.reshape(shape{w.get_shape().type(), {w0, 1, w123}});
                argument B1 = col_buffer.reshape(
                    shape{col_buffer.get_shape().type(), {w0, w123, output_image_size}});
                argument C1 = argument{shape{A.get_shape().type(), {w0, 1, output_image_size}},
                                       (int8_t*)result.data() + i * number_of_one_batch_output};
                gemm(
                    ctx, C1.get_shape(), {A1, B1, C1}, int32_t(alpha), int32_t(beta), false, false);
            }
        }
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
