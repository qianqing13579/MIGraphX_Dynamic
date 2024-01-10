#include <migraphx/gpu/device/deconvolution_2d_im2col.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/gemm_impl.hpp>
#include <migraphx/SimpleLog.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

//////////////////////////// col2img ///////////////////////////////

shape compute_col_shape2(const shape& input,
                         const shape& weights,
                         const std::vector<std::size_t>& padding,
                         const std::vector<std::size_t>& stride,
                         const std::vector<std::size_t>& dilation)
{
    shape A                       = weights;
    int number_of_one_batch_input = input.lens()[1] * input.lens()[2] * input.lens()[3];
    shape B = shape{input.type(), {A.lens()[1], number_of_one_batch_input / A.lens()[1]}};

    return {input.type(), {A.lens()[0], B.lens()[1]}};
}

template <typename DataType>
__global__ void col2im_gpu_kernel(int n,
                                  const DataType* data_col,
                                  int height,
                                  int width,
                                  int channels,
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
                                  DataType* data_im)
{
    MIGRAPHX_HIP_KERNEL_GLOBAL_STRIDE(index, n)
    {
        DataType val        = 0;
        int w_im            = index % width + pad_w;
        int h_im            = (index / width) % height + pad_h;
        int c_im            = index / (width * height);
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;

        int w_col_start = (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        int w_col_end   = min(w_im / stride_w + 1, width_col);
        int h_col_start = (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        int h_col_end   = min(h_im / stride_h + 1, height_col);

        for(int h_col = h_col_start; h_col < h_col_end; h_col += 1)
        {
            for(int w_col = w_col_start; w_col < w_col_end; w_col += 1)
            {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if(h_k % dilation_h == 0 && w_k % dilation_w == 0)
                {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int data_col_index =
                        (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) *
                            width_col +
                        w_col;
                    val += data_col[data_col_index];
                }
            }
        }
        data_im[index] = val;
    }
}

template <typename DataType>
void col2im_gpu(hipStream_t stream,
                const DataType* data_col,
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
                DataType* data_im)
{
    int height_col  = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col   = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height * width;

    col2im_gpu_kernel<DataType>
        <<<get_number_blocks(num_kernels), NUM_THREADS_PER_BLOCK, 0, stream>>>(num_kernels,
                                                                               data_col,
                                                                               height,
                                                                               width,
                                                                               channels,
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
                                                                               data_im);
}

void deconvolution_2d_im2col(context& ctx,
                             const argument& result,
                             const argument& x,
                             const argument& w,
                             const argument& col_buf,
                             const std::vector<std::size_t>& padding,
                             const std::vector<std::size_t>& stride,
                             const std::vector<std::size_t>& dilation,
                             int group,
                             bool is_1x1,
                             shape kernel_shape)
{
    int batch_size = x.get_shape().lens()[0];
    int number_of_one_batch_input =
        x.get_shape().lens()[1] * x.get_shape().lens()[2] * x.get_shape().lens()[3];
    int number_of_one_batch_output =
        result.get_shape().lens()[1] * result.get_shape().lens()[2] * result.get_shape().lens()[3];
    shape new_col_shape =
        compute_col_shape2(x.get_shape(), w.get_shape(), padding, stride, dilation);
    argument C = col_buf.reshape(new_col_shape);

    if(x.get_shape().type() == shape::float_type)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // 执行gemm计算
            argument A = w;
            argument B = argument{shape{x.get_shape().type(),
                                        {A.get_shape().lens()[1],
                                         number_of_one_batch_input / A.get_shape().lens()[1]}},
                                  (float*)x.data() + i * number_of_one_batch_input};
            ;
            float alpha = 1.0;
            float beta  = 0.0;
            gemm(ctx, C.get_shape(), {A, B, C}, alpha, beta, true, false);

            // 执行col2im
            col2im_gpu(ctx.get_stream().get(),
                       (float*)C.data(),
                       result.get_shape().lens()[1],
                       result.get_shape().lens()[2],
                       result.get_shape().lens()[3],
                       kernel_shape.lens()[2],
                       kernel_shape.lens()[3],
                       padding[0],
                       padding[1],
                       stride[0],
                       stride[1],
                       dilation[0],
                       dilation[1],
                       (float*)result.data() + i * number_of_one_batch_output);
        }
    }
    else if(x.get_shape().type() == shape::half_type)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // 执行gemm计算
            argument A = w;
            argument B = argument{shape{x.get_shape().type(),
                                        {A.get_shape().lens()[1],
                                         number_of_one_batch_input / A.get_shape().lens()[1]}},
                                  (__fp16*)x.data() + i * number_of_one_batch_input};
            ;
            float alpha = 1.0;
            float beta  = 0.0;
            gemm(ctx, C.get_shape(), {A, B, C}, alpha, beta, true, false);

            // 执行col2im
            col2im_gpu(ctx.get_stream().get(),
                       (__fp16*)C.data(),
                       result.get_shape().lens()[1],
                       result.get_shape().lens()[2],
                       result.get_shape().lens()[3],
                       kernel_shape.lens()[2],
                       kernel_shape.lens()[3],
                       padding[0],
                       padding[1],
                       stride[0],
                       stride[1],
                       dilation[0],
                       dilation[1],
                       (__fp16*)result.data() + i * number_of_one_batch_output);
        }
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
