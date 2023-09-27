#include "migraphx/gpu/device/visit.hpp"
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/upsample.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {


#define MIGRAPHX_HIP_NUM_THREADS 128

inline int MIGRAPHX_HIP_GET_BLOCKS(const int N) {
    return (N + MIGRAPHX_HIP_NUM_THREADS - 1) / MIGRAPHX_HIP_NUM_THREADS;
}

#define MIGRAPHX_HIP_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

template<typename T>
__global__ void upsample_nearest2d_kernel(int count, const T* srcData,
        T* dstData, float rheight, float rwidth, int output_c,
        int output_h, int output_w, int input_h, int input_w) {
    MIGRAPHX_HIP_KERNEL_LOOP(index, count) {
        int w = index % output_w;
        int h = (index / output_w) % output_h;
        int nc = index / output_w / output_h;
        int scaled_h = h * rheight;
        int scaled_w = w * rwidth;
        int in_index = (nc * input_h + scaled_h) * input_w + scaled_w;
        dstData[index] = srcData[in_index];
    }
}

void upsample(hipStream_t stream, const argument& result, const argument& arg, const std::vector<float> &scales,int mode,std::string coordinate_transformation_mode)
{

    // 获取输入输出大小
    migraphx::shape output_shape= result.get_shape();
    migraphx::shape input_shape= arg.get_shape();

    int count=output_shape.elements();// 所有元素个数
    int output_channel=output_shape.lens()[1];
    int output_height=output_shape.lens()[2];
    int output_width=output_shape.lens()[3];
    int input_height=input_shape.lens()[2];
    int input_width=input_shape.lens()[3];
    int num_threads=128;// 每个block的线程数

    if(coordinate_transformation_mode =="align_corners")
    {
        float rheight = (output_height > 1) ? (float)(input_height - 1) / (output_height - 1) : 0.f;
        float rwidth = (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
        hip_visit_all(result, arg)([&](auto output, auto input) 
        {
            gs_launch(stream,count,num_threads)([=](auto i, auto idx) __device__ 
            {
                int w = i % output_width;
                int h = (i / output_width) % output_height;
                int c = (i / output_width / output_height) % output_channel;
                int n = i / output_width / output_height / output_channel;
                int nc = i / output_width / output_height;

                // 计算原图坐标
                float h1r = rheight * h;
                int h1 = h1r;
                float w1r = rwidth * w;
                int w1 = w1r;

                if(mode==2) // linear，线性插值
                {
                    int h1p = (h1 < input_height - 1) ? 1 : 0;
                    float h1lambda = h1r - h1;
                    float h0lambda = (float)1. - h1lambda;
                    int w1p = (w1 < input_width - 1) ? 1 : 0;
                    float w1lambda = w1r - w1;
                    float w0lambda = (float)1. - w1lambda;
                    int left_top = (n * output_channel + c) * input_height * input_width + h1 * input_width + w1;
                    output[i] = h0lambda * (w0lambda * input[left_top] + w1lambda * input[left_top + w1p]) +
                        h1lambda * (w0lambda * input[left_top + h1p * input_width] + w1lambda * input[left_top + h1p * input_width + w1p]);

                }
                else // nearest，最近邻
                {
                    int in_index = (nc * input_height + h1) * input_width + w1;
                    output[i] = input[in_index];

                }
                
                
            }

            );

        });
    }
    else if(coordinate_transformation_mode =="pytorch_half_pixel")
    {
        float rheight = (output_height > 1) ? (float)(input_height) / output_height : 0.f;
        float rwidth = (output_width > 1) ? (float)(input_width) / output_width : 0.f;
        hip_visit_all(result, arg)([&](auto output, auto input) 
        {
            gs_launch(stream,count,num_threads)([=](auto i, auto idx) __device__ 
            {
                int w = i % output_width;
                int h = (i / output_width) % output_height;
                int c = (i / output_width / output_height) % output_channel;
                int n = i / output_width / output_height / output_channel;
                int nc = i / output_width / output_height;

                // 计算原图坐标
                float h1r = rheight * (h + 0.5) - 0.5;
                h1r = h1r >= 0 ? h1r : 0;
                int h1 = h1r;
                float w1r = rwidth * (w + 0.5) - 0.5;
                w1r = w1r >= 0? w1r : 0;
                int w1 = w1r;

                if(mode==2) // linear，线性插值
                {
                    
                    int h1p = (h1 < input_height - 1) ? 1 : 0;
                    float h1lambda = h1r - h1;
                    float h0lambda = (float)1. - h1lambda;
                    int w1p = (w1 < input_width - 1) ? 1 : 0;
                    float w1lambda = w1r - w1;
                    float w0lambda = (float)1. - w1lambda;
                    int left_top = (n * output_channel + c) * input_height * input_width + h1 * input_width + w1;
                    output[i] = h0lambda * (w0lambda * input[left_top] + w1lambda * input[left_top + w1p]) +
                        h1lambda * (w0lambda * input[left_top + h1p * input_width] + w1lambda * input[left_top + h1p * input_width + w1p]);

                }
                else // nearest，最近邻
                {
                    int in_index = (nc * input_height + h1) * input_width + w1;
                    output[i] = input[in_index];

                }
                
            }

            );

        });

    }
    else if(coordinate_transformation_mode =="half_pixel")
    {
        float rheight = (float)(input_height) / output_height;
        float rwidth = (float)(input_width) / output_width;
        hip_visit_all(result, arg)([&](auto output, auto input) 
        {
            gs_launch(stream,count,num_threads)([=](auto i, auto idx) __device__ 
            {
                int w = i % output_width;
                int h = (i / output_width) % output_height;
                int c = (i / output_width / output_height) % output_channel;
                int n = i / output_width / output_height / output_channel;
                int nc = i / output_width / output_height;

                // 计算原图坐标
                float h1r = rheight * (h + 0.5) - 0.5;
                h1r = h1r >= 0 ? h1r : 0;
                int h1 = h1r;
                float w1r = rwidth * (w + 0.5) - 0.5;
                w1r = w1r >= 0? w1r : 0;
                int w1 = w1r;

                if(mode==2) // linear，线性插值
                {
                    int h1p = (h1 < input_height - 1) ? 1 : 0;
                    float h1lambda = h1r - h1;
                    float h0lambda = (float)1. - h1lambda;
                    int w1p = (w1 < input_width - 1) ? 1 : 0;
                    float w1lambda = w1r - w1;
                    float w0lambda = (float)1. - w1lambda;

                    int left_top = (n * output_channel + c) * input_height * input_width + h1 * input_width + w1;
                    output[i] = h0lambda * (w0lambda * input[left_top] + w1lambda * input[left_top + w1p]) +
                        h1lambda * (w0lambda * input[left_top + h1p * input_width] + w1lambda * input[left_top + h1p * input_width + w1p]);
                }
                else // nearest，最近邻
                {
                    int in_index = (nc * input_height + h1) * input_width + w1;
                    output[i] = input[in_index];

                }
            }

            );

        });

    }
    else if(coordinate_transformation_mode =="asymmetric")
    {
        float rheight = (float)(input_height) / output_height;
        float rwidth = (float)(input_width) / output_width;
        hip_visit_all(result, arg)([&](auto output, auto input) 
        {
            gs_launch(stream,count,num_threads)([=](auto i, auto idx) __device__ 
            {
                int w = i % output_width;
                int h = (i / output_width) % output_height;
                int c = (i / output_width / output_height) % output_channel;
                int n = i / output_width / output_height / output_channel;
                int nc = i / output_width / output_height;

                // 计算原图坐标
                float h1r = rheight * h;
                h1r = h1r >= 0 ? h1r : 0;
                int h1 = h1r;
                float w1r = rwidth * w;
                w1r = w1r >= 0? w1r : 0;
                int w1 = w1r;

                if(mode==2) // linear，线性插值
                {
                    int h1p = (h1 < input_height - 1) ? 1 : 0;
                    float h1lambda = h1r - h1;
                    float h0lambda = (float)1. - h1lambda;
                    int w1p = (w1 < input_width - 1) ? 1 : 0;
                    float w1lambda = w1r - w1;
                    float w0lambda = (float)1. - w1lambda;
                    int left_top = (n * output_channel + c) * input_height * input_width + h1 * input_width + w1;
                    output[i] = h0lambda * (w0lambda * input[left_top] + w1lambda * input[left_top + w1p]) +
                        h1lambda * (w0lambda * input[left_top + h1p * input_width] + w1lambda * input[left_top + h1p * input_width + w1p]);

                }
                else // nearest，最近邻
                {
                    int in_index = (nc * input_height + h1) * input_width + w1;
                    output[i] = input[in_index];

                }

                
            }

            );

        });

    }

    // 使用通用kernel写法，通用kernel存在的问题是由于无法使用tensor_view访问，所以只能使用standard shape数据
    // if(arg.get_shape().type()==shape::float_type)
    // {
    //     upsample_nearest2d_kernel<<<MIGRAPHX_HIP_GET_BLOCKS(count), MIGRAPHX_HIP_NUM_THREADS, 0, stream>>>(count,
    //         (float*)arg.data(), (float*)result.data(), rheight, rwidth, output_channel, output_height, output_width, input_height, input_width);

    // }
    // else if(arg.get_shape().type()==shape::half_type)
    // {
    //     upsample_nearest2d_kernel<<<MIGRAPHX_HIP_GET_BLOCKS(count), MIGRAPHX_HIP_NUM_THREADS, 0, stream>>>(count,
    //         (_Float16*)arg.data(), (_Float16*)result.data(), rheight, rwidth, output_channel, output_height, output_width, input_height, input_width);

    // }
    


    
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
