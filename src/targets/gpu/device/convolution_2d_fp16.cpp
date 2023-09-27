/**
 * 该FP16的实现在906以后的版本才支持，该实现包含了906不支持的指令
 * 
 * 
 */

#include <migraphx/gpu/device/convolution_2d_fp16.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// typedef long BB __attribute__((ext_vector_type(2)));

// #define depthU_C 32
// #define T_M 4
// #define T_M_bit 2
// #define T_N 4
// #define T_N_bit 2
// #define THREADS_NUM_M 16
// #define THREADS_NUM_N 16
// #define READ_A_Num 8
// #define READ_A_ROW_bit 2
// #define READ_A_COL 3

// #define READ_B_Num 4
// #define READ_B_COL_bit 2
// #define READ_B_COL 3

// #define GetBit(x,y) ((x) >> (y)&1)	//获取将x的第y位（0或1）
// #define SetBit(x,y) x|=(1<<y) 		//将X的第Y位置1
// #define ClrBit(x,y) x&=~(1<<y) 		//将X的第Y位清0



// #define LDS_SIZE      1024
// #define LDS_OFFSET    1024


// typedef struct fp16_4
// {
//     _Float16 x;
//     _Float16 y;
//     _Float16 z;
//     _Float16 w;
// }fp16_4;



// #define MAC_4X4_F16 \
// C0.x += A.x*B.x;\
// C0.y += A.x*B.y;\
// C0.z += A.x*B.z;\
// C0.w += A.x*B.w;\
// C1.x += A.y*B.x;\
// C1.y += A.y*B.y;\
// C1.z += A.y*B.z;\
// C1.w += A.y*B.w;\
// C2.x += A.z*B.x;\
// C2.y += A.z*B.y;\
// C2.z += A.z*B.z;\
// C2.w += A.z*B.w;\
// C3.x += A.w*B.x;\
// C3.y += A.w*B.y;\
// C3.z += A.w*B.z;\
// C3.w += A.w*B.w;\

// #define MAC_4X4_F16_1  \
// C0.x += A1.x*B1.x;\
// C0.y += A1.x*B1.y;\
// C0.z += A1.x*B1.z;\
// C0.w += A1.x*B1.w;\
// C1.x += A1.y*B1.x;\
// C1.y += A1.y*B1.y;\
// C1.z += A1.y*B1.z;\
// C1.w += A1.y*B1.w;\
// C2.x += A1.z*B1.x;\
// C2.y += A1.z*B1.y;\
// C2.z += A1.z*B1.z;\
// C2.w += A1.z*B1.w;\
// C3.x += A1.w*B1.x;\
// C3.y += A1.w*B1.y;\
// C3.z += A1.w*B1.z;\
// C3.w += A1.w*B1.w;\



//  //4x4 4x4 性能比4x4 8x2性能好
// #define LDS_READ_AB(times) \
// B = *((int4 *)(&ldsb[lrb+ 2*((times>>2)*64+(times&3)*4)])); \
// A = *((int4 *)(&ldsa[lra+ times*64+0])); \


// #define LDS_READ_A1B1(times) \
// B1 = *((int4 *)(&ldsb[lrb+ 2*((times>>2)*64+(times&3)*4)])); \
// A1 = *((int4 *)(&ldsa[lra+ times*64+0])); \

// #define LDS_READ_AB_HALF(times) \
// B = *((int4 *)(&ldsb[lrb+ 2*((times>>2)*64+(times&3)*4) + 1024])); \
// A = *((int4 *)(&ldsa[lra+ (times + 8)*64+0])); \


// #define LDS_READ_A1B1_HALF(times) \
// B1 = *((int4 *)(&ldsb[lrb+ 2*((times>>2)*64+(times&3)*4) + 1024])); \
// A1 = *((int4 *)(&ldsa[lra+ (times + 8)*64+0])); \



// //B x-8x2
// // #define LDS_READ_AB(times) \
// // B = *((float4 *)(&ldsb[lrb+ 2*((times>>3)*128+(times&7)*4)])); \
// // A = *((float4 *)(&ldsa[lra+ times*64+0])); \


// // #define LDS_READ_A1B1(times) \
// // B1 = *((float4 *)(&ldsb[lrb+ 2*((times>>3)*128+(times&7)*4)])); \
// // A1 = *((float4 *)(&ldsa[lra+ times*64+0])); \

// // #define LDS_READ_AB_HALF(times) \
// // B = *((float4 *)(&ldsb[lrb+ 2*((times>>3)*128+(times&7)*4) + 1024])); \
// // A = *((float4 *)(&ldsa[lra+ (times + 8)*64+0])); \


// // #define LDS_READ_A1B1_HALF(times) \
// // B1 = *((float4 *)(&ldsb[lrb+ 2*((times>>3)*128+(times&7)*4) + 1024])); \
// // A1 = *((float4 *)(&ldsa[lra+ (times + 8)*64+0])); \




// __attribute__((device)) static inline void bufferloadA(int4 *gA, int* offsetA0,int* offsetA1, int* offsetA2,int* offsetA3,BB* globalReadA)
// {
//     // asm volatile( 
                                    

//     //             "buffer_load_dword %0,%4,%8,0, idxen\n"
//     //             "buffer_load_dword %1,%5,%8,0, idxen\n"               
//     //             "buffer_load_dword %2,%6,%8,0, idxen\n"                
//     //             "buffer_load_dword %3,%7,%8,0, idxen\n"

//     //              "s_waitcnt vmcnt(0)\n\t"

//     //             :"=v"(gA->x),"=v"(gA->y),"=v"(gA->z),"=v"(gA->w),"+v"(*offsetA0), "+v"(*offsetA1),"+v"(*offsetA2),"+v"(*offsetA3),"+s"(*globalReadA));
// }
//  __attribute__((device)) static inline void bufferloadB(fp16_4 *gB, int* offsetB0,int* offsetB1, int* offsetB2,int* offsetB3,BB* globalReadB)
// {
//     // asm volatile( 

//     //             "buffer_load_short_d16 %0,%4,%8,0, idxen\n"
//     //             "buffer_load_short_d16 %1,%5,%8,0, idxen\n"               
//     //             "buffer_load_short_d16 %2,%6,%8,0, idxen\n"              
//     //             "buffer_load_short_d16 %3,%7,%8,0, idxen\n"

//     //             "s_waitcnt vmcnt(0)\n\t"
                 
//     //             :"=v"(gB->x),"=v"(gB->y),"=v"(gB->z),"=v"(gB->w),"+v"(*offsetB0), "+v"(*offsetB1),"+v"(*offsetB2),"+v"(*offsetB3),"+s"(*globalReadB));
// } 

// __attribute__((device)) static inline void Dot2_F32_F16(int4 *A, int4 *B,float4 *C0,float4 *C1,float4 *C2,float4 *C3)
// {
//     // asm volatile(

//     //               "v_dot2_f32_f16 %8,%0,%4,%8, \n"
//     //               "v_dot2_f32_f16 %9,%0,%5,%9, \n"
//     //               "v_dot2_f32_f16 %10,%0,%6,%10,\n"
//     //               "v_dot2_f32_f16 %11,%0,%7,%11, \n"

//     //               "v_dot2_f32_f16 %12,%1,%4,%12, \n"
//     //               "v_dot2_f32_f16 %13,%1,%5,%13, \n"
//     //               "v_dot2_f32_f16 %14,%1,%6,%14, \n"
//     //               "v_dot2_f32_f16 %15,%1,%7,%15, \n"

//     //               "v_dot2_f32_f16 %16,%2,%4,%16, \n"
//     //               "v_dot2_f32_f16 %17,%2,%5,%17,\n"
//     //               "v_dot2_f32_f16 %18,%2,%6,%18, \n"
//     //               "v_dot2_f32_f16 %19,%2,%7,%19, \n"
 
//     //               "v_dot2_f32_f16 %20,%3,%4,%20, \n"
//     //               "v_dot2_f32_f16 %21,%3,%5,%21, \n"
//     //               "v_dot2_f32_f16 %22,%3,%6,%22, \n"
//     //               "v_dot2_f32_f16 %23,%3,%7,%23, \n"


//     //              :"+v"(A->x), "+v"(A->y),"+v"(A->z),"+v"(A->w),"+v"(B->x), "+v"(B->y),"+v"(B->z),"+v"(B->w),"+v"(C0->x),"+v"(C0->y),"+v"(C0->z),"+v"(C0->w),"+v"(C1->x),"+v"(C1->y),"+v"(C1->z),"+v"(C1->w),"+v"(C2->x),"+v"(C2->y),"+v"(C2->z),"+v"(C2->w),"+v"(C3->x),"+v"(C3->y),"+v"(C3->z),"+v"(C3->w));

// }
// __attribute__((device)) static inline void Dot2_F32_F16_CLear_A_HI(int4 *A, int4 *B,float4 *C0,float4 *C1,float4 *C2,float4 *C3)
// {

//     //              A->x = A->x & 0x0000FFFF;
//     //              A->y = A->y & 0x0000FFFF;
//     //              A->z = A->z & 0x0000FFFF;
//     //              A->w = A->w & 0x0000FFFF;

//     // asm volatile(

//     //               "v_dot2_f32_f16 %8,%0,%4,%8, \n"
//     //               "v_dot2_f32_f16 %9,%0,%5,%9, \n"
//     //               "v_dot2_f32_f16 %10,%0,%6,%10,\n"
//     //               "v_dot2_f32_f16 %11,%0,%7,%11, \n"

//     //               "v_dot2_f32_f16 %12,%1,%4,%12, \n"
//     //               "v_dot2_f32_f16 %13,%1,%5,%13, \n"
//     //               "v_dot2_f32_f16 %14,%1,%6,%14, \n"
//     //               "v_dot2_f32_f16 %15,%1,%7,%15, \n"

//     //               "v_dot2_f32_f16 %16,%2,%4,%16, \n"
//     //               "v_dot2_f32_f16 %17,%2,%5,%17,\n"
//     //               "v_dot2_f32_f16 %18,%2,%6,%18, \n"
//     //               "v_dot2_f32_f16 %19,%2,%7,%19, \n"
 
//     //               "v_dot2_f32_f16 %20,%3,%4,%20, \n"
//     //               "v_dot2_f32_f16 %21,%3,%5,%21, \n"
//     //               "v_dot2_f32_f16 %22,%3,%6,%22, \n"
//     //               "v_dot2_f32_f16 %23,%3,%7,%23, \n"


//     //              :"+v"(A->x), "+v"(A->y),"+v"(A->z),"+v"(A->w),"+v"(B->x), "+v"(B->y),"+v"(B->z),"+v"(B->w),"+v"(C0->x),"+v"(C0->y),"+v"(C0->z),"+v"(C0->w),"+v"(C1->x),"+v"(C1->y),"+v"(C1->z),"+v"(C1->w),"+v"(C2->x),"+v"(C2->y),"+v"(C2->z),"+v"(C2->w),"+v"(C3->x),"+v"(C3->y),"+v"(C3->z),"+v"(C3->w));

// }


// __attribute__((device)) static inline void V_Cvt_F16_F32(float4 *C0,float4 *C1,float4 *C2,float4 *C3)
// {
//     // asm volatile(
//     //               "v_cvt_f16_f32 %0,%0 \n" 
//     //               "v_cvt_f16_f32 %1,%1 \n" 
//     //               "v_cvt_f16_f32 %2,%2 \n" 
//     //               "v_cvt_f16_f32 %3,%3 \n" 

//     //               "v_cvt_f16_f32 %4,%4 \n" 
//     //               "v_cvt_f16_f32 %5,%5 \n" 
//     //               "v_cvt_f16_f32 %6,%6 \n" 
//     //               "v_cvt_f16_f32 %7,%7 \n" 

//     //               "v_cvt_f16_f32 %8,%8 \n" 
//     //               "v_cvt_f16_f32 %9,%9 \n" 
//     //               "v_cvt_f16_f32 %10,%10 \n" 
//     //               "v_cvt_f16_f32 %11,%11 \n" 
 
//     //               "v_cvt_f16_f32 %12,%12 \n" 
//     //               "v_cvt_f16_f32 %13,%13 \n" 
//     //               "v_cvt_f16_f32 %14,%14 \n" 
//     //               "v_cvt_f16_f32 %15,%15 \n" 

//     //              :"+v"(C0->x),"+v"(C0->y),"+v"(C0->z),"+v"(C0->w),"+v"(C1->x),"+v"(C1->y),"+v"(C1->z),"+v"(C1->w),"+v"(C2->x),"+v"(C2->y),"+v"(C2->z),"+v"(C2->w),"+v"(C3->x),"+v"(C3->y),"+v"(C3->z),"+v"(C3->w));

// }
// __attribute__((device)) static inline void V_cvt_pkrtz_f16_f32(float2 *D,float4 *C)
// {
//     // asm volatile( 
                                    
//     //             "v_cvt_pkrtz_f16_f32 %0,%2,%3\n"
//     //             "v_cvt_pkrtz_f16_f32 %1,%4,%5\n"                                     
//     //             :"+v"(D->x),"+v"(D->y),"+v"(C->x),"+v"(C->y),"+v"(C->z),"+v"(C->w));
// } 
// __attribute__((device)) static inline void bufferstoreC(float2 *D, int* offsetC0,int* offsetC1, int* offsetC2,int* offsetC3,BB* globalStoreC)
// {
//     asm volatile( 
                                    
//                 "buffer_store_short %0,%2,%6,0, idxen\n"
//                 "buffer_store_short_d16_hi %0,%3,%6,0, idxen\n"                 
//                 "buffer_store_short %1,%4,%6,0, idxen\n"                 
//                 "buffer_store_short_d16_hi %1,%5,%6,0, idxen\n"
                    
//                 "s_waitcnt vmcnt(0)\n\t"
//                 :"+v"(D->x),"+v"(D->y),"+v"(*offsetC0), "+v"(*offsetC1),"+v"(*offsetC2),"+v"(*offsetC3),"+s"(*globalStoreC));
// } 



// union FP32
// {
//     uint u;
//     float f;
//     struct
//     {
//         uint Mantissa : 23;
//         uint Exponent : 8;
//         uint Sign : 1;
//     };
// };

// union FP16
// {
//     unsigned short u;
//     struct
//     {
//         uint Mantissa : 10;
//         uint Exponent : 5;
//         uint Sign : 1;
//     };
// };




// //加汇编-1读取
// __global__ void implicit_gemmComAlg(_Float16 * gemmA,
//                                              _Float16 * gemmB,
//                                              _Float16 * gemmC,
//                                             unsigned int strideA_0,
//                                             unsigned int strideA_1,
//                                             unsigned int strideA_2,
//                                             unsigned int strideA_3,
//                                             unsigned int strideA_4,
//                                             unsigned int strideB_0,
//                                             unsigned int strideB_1,
//                                             unsigned int strideB_2,
//                                             unsigned int strideB_3,
//                                             unsigned int strideB_4,
//                                             unsigned int strideC_0,
//                                             unsigned int strideC_1,
//                                             unsigned int strideC_2,
//                                             unsigned int strideC_3,
//                                             unsigned int last_k_res_mask_bit,
//                                             unsigned int last_k_res,
//                                             unsigned int stride_H,
//                                             unsigned int stride_W,
//                                             unsigned int pad_H,
//                                             unsigned int pad_W,
//                                             unsigned int dilate_H,
//                                             unsigned int dilate_W,
//                                             unsigned int group,
//                                             unsigned int group_out_channels,
//                                             unsigned int group_block_nums,
//                                             unsigned int numgroupX,
//                                             unsigned int numgroupY,
//                                             unsigned int remdinergroupX,
//                                             unsigned int remdinergroupY)
// {

//          BB globalReadA;
//          globalReadA.x = (long)gemmA;
//          globalReadA.x = (globalReadA.x | (((long)(0x2 << 16))<<32)); //const_stride    
//          globalReadA.y = (((((long)0x20000)<<32) | 0xFFFFFFFE));
//          BB globalReadB;
//          globalReadB.x = (long)gemmB;
//          globalReadB.x = (globalReadB.x | (((long)(0x2 << 16))<<32)); //const_stride      
//          globalReadB.y = (((((long)0x20000)<<32) | 0xFFFFFFFE));
//          BB globalStoreC;
//          globalStoreC.x = (long)gemmC;
//          globalStoreC.x = (globalStoreC.x | (((long)(0x2 << 16))<<32)); //const_stride   
//          globalStoreC.y = (((((long)0x20000)<<32) | 0xFFFFFFFE));

//          BB globalReadA1 = globalReadA;
//          BB globalReadB1 = globalReadB;
//          BB globalStoreC1 = globalStoreC;

//         int threadidInblock =  threadIdx.x + threadIdx.y*blockDim.x;
//         int readcol = threadidInblock & READ_A_COL;
//         int readrow = threadidInblock >> READ_A_ROW_bit ;
        
//         // int readcol = threadidInblock >> 6;
//         // int readrow = threadidInblock & 63;

//         // 4x4 4x4 性能比4x4 8x2性能好
//          unsigned int lwa = readrow + readcol * (READ_A_Num>>1) * 64;
//          unsigned int lwb = 2*((threadIdx.y >>2)*128  + (threadIdx.y&3) * 16 + (threadIdx.x >>3)*64 +((threadIdx.x>>1) & 3)*4) + (threadIdx.x & 1);

//         //  unsigned int lwb = 2*((threadIdx.y >>2)*128  + (threadIdx.y&3) * 32 + (threadIdx.x >>4)*128 +((threadIdx.x>>1) & 7)*4) + (threadIdx.x & 1);
//         //  unsigned int lrb = 2*((threadIdx.y >>2)*128  + (threadIdx.y&3) * 32);
      
//          unsigned int lra = 4 * threadIdx.x ;
//          unsigned int lrb = 2*((threadIdx.y >>2)*128  + (threadIdx.y&3) * 16);
    
//         //GEMM
//          __shared__  int ldsa[LDS_SIZE];
//          __shared__  _Float16 ldsb[LDS_SIZE*2];

          
//          int4 gA;
//          fp16_4 gB;
//          fp16_4 gB1;

//          int4 A;
//          int4 B;
//          int4 A1;
//          int4 B1;
 
//         //控制边界
//         //Gemm 到Conv 坐标转换
//         //dst_c
//         unsigned int g = blockIdx.z % group;
//         unsigned int oc = g *group_out_channels + (64 *  blockIdx.x) + readrow ;  
//         unsigned int n =  (blockIdx.z/group) * strideB_0*group  + g * strideB_0;
//         unsigned int j_res = 64 * blockIdx.y  + (threadIdx.y) * 4;
//         //src_b 
//         int oh0;
//         int ow0;
//         int oh1;
//         int ow1;
//         int oh2;
//         int ow2;
//         int oh3;
//         int ow3;
//         //src_a
//         unsigned int ic = (threadIdx.x) /(strideA_1);
//         unsigned int k_res =(threadIdx.x) %(strideA_1);
//         unsigned int fh  = (k_res / strideA_2) * dilate_H;
//         unsigned int fw  = (k_res % strideA_2) * dilate_W;
//         //step_k
//         //unsigned int step_k = 0;
//         unsigned int tem = 0;
//         unsigned int tem1 = 0;

//         int offset0_tmp;
//         int offset1_tmp;
//         int offset2_tmp;
//         int offset3_tmp;


//         // gA.x = 1.0;
//         // gA.y = 1.0;
//         // gA.z = 1.0;
//         // gA.w = 1.0;
//         // gB.x = 1.0;
//         // gB.y = 1.0;
//         // gB.z = 1.0;
//         // gB.w = 1.0;
//          //global->lds  

//         //B
//          oh0 = ((j_res + 0) / strideC_2) * stride_H - pad_H ;
//          ow0 = ((j_res + 0) % strideC_2) * stride_W - pad_W;

//          oh1 = ((j_res + 1) / strideC_2) * stride_H - pad_H;
//          ow1 = ((j_res + 1) % strideC_2) * stride_W - pad_W;

//          oh2 = ((j_res + 2) / strideC_2) * stride_H - pad_H;
//          ow2 = ((j_res + 2) % strideC_2) * stride_W - pad_W;

//          oh3 = ((j_res + 3) / strideC_2) * stride_H - pad_H;
//          ow3 = ((j_res + 3) % strideC_2) * stride_W - pad_W;

  
//         tem = (n  + ic * strideB_1 + fh * strideB_2 + fw) ; 

//         int offset0 = (oh0 *strideB_2 +  ow0);       
//         int offset1 = (oh1 *strideB_2 +  ow1);
//         int offset2 = (oh2 *strideB_2 +  ow2);
//         int offset3 = (oh3 *strideB_2 +  ow3);

//         offset0_tmp =(((fh  + oh0) < strideB_3 && fw  + ow0 < strideB_2) &&  (threadIdx.x < strideA_0))?(tem + offset0):-1;
//         offset1_tmp =(((fh  + oh1) < strideB_3 && fw  + ow1 < strideB_2) &&  (threadIdx.x < strideA_0))?(tem + offset1):-1;
//         offset2_tmp =(((fh  + oh2) < strideB_3 && fw  + ow2 < strideB_2) &&  (threadIdx.x < strideA_0))?(tem + offset2):-1;
//         offset3_tmp =(((fh  + oh3) < strideB_3 && fw  + ow3 < strideB_2) &&  (threadIdx.x < strideA_0))?(tem + offset3):-1;
       
//         bufferloadB(&gB, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadB1);

     
//         ic = (threadIdx.x + 16) /(strideA_1);
//         k_res =(threadIdx.x + 16) %(strideA_1);
//         fh  = (k_res / strideA_2) * dilate_H;
//         fw  = (k_res % strideA_2) * dilate_W;

//         tem =(n  + ic * strideB_1 + fh * strideB_2 + fw) ;    

//         offset0_tmp =(((fh  + oh0) < strideB_3 && fw  + ow0 < strideB_2) &&  (threadIdx.x + 16< strideA_0))?(tem + offset0):-1;
//         offset1_tmp =(((fh  + oh1) < strideB_3 && fw  + ow1 < strideB_2) &&  (threadIdx.x + 16< strideA_0))?(tem + offset1):-1;
//         offset2_tmp =(((fh  + oh2) < strideB_3 && fw  + ow2 < strideB_2) &&  (threadIdx.x + 16< strideA_0))?(tem + offset2):-1;
//         offset3_tmp =(((fh  + oh3) < strideB_3 && fw  + ow3 < strideB_2) &&  (threadIdx.x + 16< strideA_0))?(tem + offset3):-1;

//         bufferloadB(&gB1, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadB1);
        
//         //gB.w = 1.0;
//          //A     
//          unsigned int A_edge = strideA_4 < (g + 1)*group_out_channels * strideA_0 ? strideA_4 : (g + 1)*group_out_channels * strideA_0;
//          int base_A = (oc * strideA_0  + 0 + readcol*READ_A_Num);
//          offset0_tmp =(base_A < A_edge )?((0 + base_A)):-1;

//          asm volatile(
//                 "buffer_load_dwordx4 %0,%1,%2,0, idxen \n"
//                  "s_waitcnt vmcnt(0)\n\t"
//             :"=v"(gA), "+v"(offset0_tmp), "+s"(globalReadA1));  

//         // _Float16 *gax1=(_Float16 *)(&(gA.x));
//         // printf("turn  blockIdx.x %d blockIdx.y %d blockIdx.z %d threadIdx.x %d threadIdx.y %d gA.x1 %f gA.x2 %f\n",blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,(float)*gax1,(float)*(gax1+1));
//         // _Float16 *gay1=(_Float16 *)(&(gA.y));
//         // printf("turn  blockIdx.x %d blockIdx.y %d blockIdx.z %d threadIdx.x %d threadIdx.y %d gA.y1 %f gA.y2 %f\n",blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,(float)*gay1,(float)*(gay1+1));
//         // _Float16 *gaz1=(_Float16 *)(&(gA.z));
//         // printf("turn  blockIdx.x %d blockIdx.y %d blockIdx.z %d threadIdx.x %d threadIdx.y %d gA.z1 %f gA.z2 %f\n",blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,(float)*gaz1,(float)*(gaz1+1));
//         // _Float16 *gaw1=(_Float16 *)(&(gA.w));
//         // printf("turn  blockIdx.x %d blockIdx.y %d blockIdx.z %d threadIdx.x %d threadIdx.y %d gA.w1 %f gA.w2 %f\n",blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,(float)*gaw1,(float)*(gaw1+1));


//          //bufferloadA(&gA, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadA1);
//         // if((1 + base_A) >= A_edge )
//         // {
//         //      gA.x = gA.x & 0x0000FFFF;
//         // }
//         // if((3 + base_A) >= A_edge )
//         // {
//         //      gA.y = gA.y & 0x0000FFFF;
//         // }
//         // if((5 + base_A) >= A_edge )
//         // {
//         //      gA.z = gA.z & 0x0000FFFF;
//         // }
//         // if((7 + base_A) >= A_edge )
//         // {
//         //      gA.w = gA.w & 0x0000FFFF;
//         // }

//        //__syncthreads();

//         //write lds
//         ldsa[lwa + 0*64] = gA.x;
//         ldsa[lwa + 1*64] = gA.y;
//         ldsa[lwa + 2*64] = gA.z;
//         ldsa[lwa + 3*64] = gA.w;

//         ldsb[lwb + 0] = gB.x;
//         ldsb[lwb + 2] = gB.y;
//         ldsb[lwb + 4] = gB.z;
//         ldsb[lwb + 6] = gB.w;
//         ldsb[lwb + 0 + 1024] = gB1.x;
//         ldsb[lwb + 2 + 1024] = gB1.y;
//         ldsb[lwb + 4 + 1024] = gB1.z;
//         ldsb[lwb + 6 + 1024] = gB1.w;
     

//         // lwa = (lwa + LDS_OFFSET) & 2047;
//         // lwb = (lwb + LDS_OFFSET) & 2047;

//          __syncthreads();
//         //其他计算
//          float4 C0 ;
//          float4 C1 ;
//          float4 C2 ;
//          float4 C3 ;
//          float2 D;
//          C0.x = 0;
//          C0.y = 0;
//          C0.z = 0;
//          C0.w = 0;
//          C1.x = 0;
//          C1.y = 0;
//          C1.z = 0;
//          C1.w = 0;
//          C2.x = 0;
//          C2.y = 0;
//          C2.z = 0;
//          C2.w = 0;
//          C3.x = 0;
//          C3.y = 0;
//          C3.z = 0;
//          C3.w = 0;

//         //preload
//          LDS_READ_AB(0)
//          tem1 = threadIdx.x + depthU_C;   
   
//          for( unsigned int i = 0;i < strideA_0 - last_k_res;i+=depthU_C)
//          {

//              //preload 
//              //inter0
//              LDS_READ_A1B1(1)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
            
//              //inter1
//              LDS_READ_AB(2)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);

//              //inter2
//              LDS_READ_A1B1(3)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);

             
//              //inter3
//              LDS_READ_AB(4)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);

                          
//              //inter4
//              LDS_READ_A1B1(5)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);           

//              //inter5
//              LDS_READ_AB(6)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);

            
//             //inter6
//              LDS_READ_A1B1(7)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
          

//              //inter7
//              LDS_READ_AB_HALF(0)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);

//             //inter8
//              LDS_READ_A1B1_HALF(1)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
            
//              //inter9
//              LDS_READ_AB_HALF(2)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);


//             //inter10
//              LDS_READ_A1B1_HALF(3)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);

//              //inter11
//              LDS_READ_AB_HALF(4)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);

//              //inter12
//              LDS_READ_A1B1_HALF(5)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);

//              //inter13
//              LDS_READ_AB_HALF(6)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);

//              //inter14
//              LDS_READ_A1B1_HALF(7)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
        

//             //global->lds
//             //read_row_flag = 0;
//             ic = (i + tem1) / strideA_1;
//             k_res=(i + tem1) % strideA_1;
//             fh = (k_res / strideA_2)* dilate_H;
//             fw = (k_res % strideA_2)* dilate_W;

//         // gA.x = 1.0;
//         // gA.y = 1.0;
//         // gA.z = 1.0;
//         // gA.w = 1.0;
//         // gB.x = 1.0;
//         // gB.y = 1.0;
//         // gB.z = 1.0;
//         // gB.w = 1.0;
//              //B
//             // tem =2*(n  + ic * strideB_1 + fh * strideB_2 + fw) ;         
//             // offset0_tmp =(((fh  + oh0) < strideB_3 && fw  + ow0 < strideB_2 ) && i + tem1  < strideA_0)?(tem + offset0):-1;
//             // offset1_tmp =(((fh  + oh1) < strideB_3 && fw  + ow1 < strideB_2 ) && i + tem1  < strideA_0)?(tem + offset1):-1;
//             // offset2_tmp =(((fh  + oh2) < strideB_3 && fw  + ow2 < strideB_2 ) && i + tem1  < strideA_0)?(tem + offset2):-1;
//             // offset3_tmp =(((fh  + oh3) < strideB_3 && fw  + ow3 < strideB_2 ) && i + tem1  < strideA_0)?(tem + offset3):-1;                              
//             // bufferloadB(&gB, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadB);

//             tem = n  + ic * strideB_1;
//             offset0_tmp =(((fh  + oh0) < strideB_3 && fw  + ow0 < strideB_2 ) && i + tem1  < strideA_0)?( ( tem + (fh  + oh0) * strideB_2 + fw  + ow0)):-1;
//             offset1_tmp =(((fh  + oh1) < strideB_3 && fw  + ow1 < strideB_2 ) && i + tem1  < strideA_0)?( ( tem + (fh  + oh1) * strideB_2 + fw  + ow1)):-1;
//             offset2_tmp =(((fh  + oh2) < strideB_3 && fw  + ow2 < strideB_2 ) && i + tem1  < strideA_0)?( ( tem + (fh  + oh2) * strideB_2 + fw  + ow2)):-1;
//             offset3_tmp =(((fh  + oh3) < strideB_3 && fw  + ow3 < strideB_2 ) && i + tem1  < strideA_0)?( ( tem + (fh  + oh3) * strideB_2 + fw  + ow3)):-1;                                        
//             bufferloadB(&gB, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadB);

//             //asm volatile("s_waitcnt vmcnt(0)\n\t");

//             ic = (i + tem1 + 16) / strideA_1;
//             k_res=(i + tem1 + 16) % strideA_1;
//             fh = (k_res / strideA_2)* dilate_H;
//             fw = (k_res % strideA_2)* dilate_W;

//             // tem =2*(n  + ic * strideB_1 + fh * strideB_2 + fw) ;            
//             // offset0_tmp =(((fh  + oh0) < strideB_3 && fw  + ow0 < strideB_2 ) && i + tem1 + 16 < strideA_0)?(tem + offset0):-1;
//             // offset1_tmp =(((fh  + oh1) < strideB_3 && fw  + ow1 < strideB_2 ) && i + tem1 + 16 < strideA_0)?(tem + offset1):-1;
//             // offset2_tmp =(((fh  + oh2) < strideB_3 && fw  + ow2 < strideB_2 ) && i + tem1 + 16 < strideA_0)?(tem + offset2):-1;
//             // offset3_tmp =(((fh  + oh3) < strideB_3 && fw  + ow3 < strideB_2 ) && i + tem1 + 16 < strideA_0)?(tem + offset3):-1;             
//             // bufferloadB(&gB1, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadB);

//             tem = n  + ic * strideB_1;
//             offset0_tmp =(((fh  + oh0) < strideB_3 && fw  + ow0 < strideB_2 ) && i + tem1 + 16 < strideA_0)?( ( tem + (fh  + oh0) * strideB_2 + fw  + ow0)):-1;
//             offset1_tmp =(((fh  + oh1) < strideB_3 && fw  + ow1 < strideB_2 ) && i + tem1 + 16 < strideA_0)?( ( tem + (fh  + oh1) * strideB_2 + fw  + ow1)):-1;
//             offset2_tmp =(((fh  + oh2) < strideB_3 && fw  + ow2 < strideB_2 ) && i + tem1 + 16 < strideA_0)?( ( tem + (fh  + oh2) * strideB_2 + fw  + ow2)):-1;
//             offset3_tmp =(((fh  + oh3) < strideB_3 && fw  + ow3 < strideB_2 ) && i + tem1 + 16 < strideA_0)?( ( tem + (fh  + oh3) * strideB_2 + fw  + ow3)):-1;                                        
//             bufferloadB(&gB1, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadB);

//             //asm volatile("s_waitcnt vmcnt(0)\n\t");
//             //A
//             tem = (base_A + i + depthU_C);
//             offset0_tmp =((0 + tem) < A_edge )?((0 + tem)):-1;

//              asm volatile(
//                 "buffer_load_dwordx4 %0,%1,%2,0, idxen \n"
//                 "s_waitcnt vmcnt(0)\n\t"
//             :"=v"(gA), "+v"(offset0_tmp), "+s"(globalReadA));

//              //bufferloadA(&gA, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadA);
            
//             //write lds    
//             // lwa = (lwa + LDS_OFFSET) & 2047;
//             // lwb = (lwb + 2*LDS_OFFSET) & 4095;
//             // lra = (lra + LDS_OFFSET) & 2047;
//             // lrb = (lrb + 2*LDS_OFFSET) & 4095;

        
//             // if((1 + tem) >= A_edge)
//             // {
//             //      gA.x = gA.x & 0x0000FFFF;
//             // }
//             // if((3 + tem) >= A_edge)
//             // {
//             //      gA.y = gA.y & 0x0000FFFF;
//             // }
//             // if((5 + tem) >= A_edge)
//             // {
//             //      gA.z = gA.z & 0x0000FFFF;
//             // }
//             // if((7 + tem) >= A_edge)
//             // {
//             //      gA.w = gA.w & 0x0000FFFF;
//             // }



//             //asm volatile("s_waitcnt vmcnt(0)\n\t");
//             __syncthreads();

//             ldsa[lwa + 0*64] = gA.x;
//             ldsa[lwa + 1*64] = gA.y;
//             ldsa[lwa + 2*64] = gA.z;
//             ldsa[lwa + 3*64] = gA.w;
//             ldsb[lwb + 0] = gB.x;
//             ldsb[lwb + 2] = gB.y;
//             ldsb[lwb + 4] = gB.z;
//             ldsb[lwb + 6] = gB.w;
//             ldsb[lwb + 0 + 1024] = gB1.x;
//             ldsb[lwb + 2 + 1024] = gB1.y;
//             ldsb[lwb + 4 + 1024] = gB1.z;
//             ldsb[lwb + 6 + 1024] = gB1.w;
         
//              __syncthreads();
//             //inter15 //preload
    
//              LDS_READ_AB(0)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);   
             
//          }
                   
//             //last inter
//             //inter0
//              if(GetBit(last_k_res_mask_bit,0) && GetBit(last_k_res_mask_bit,1))
//              {
//                  LDS_READ_A1B1(1)
//                  Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,0) && !GetBit(last_k_res_mask_bit,1))
//              {
//                  LDS_READ_A1B1(1)
//                  Dot2_F32_F16_CLear_A_HI(&A,&B,&C0,&C1,&C2,&C3);
//              }
            
//              //inter1
//              if(GetBit(last_k_res_mask_bit,2) && GetBit(last_k_res_mask_bit,3))
//              {
//               LDS_READ_AB(2)
//               Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,2) && !GetBit(last_k_res_mask_bit,3))
//              {
//                LDS_READ_AB(2)
//                Dot2_F32_F16_CLear_A_HI(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
                  
//             //inter2
//             if(GetBit(last_k_res_mask_bit,4) && GetBit(last_k_res_mask_bit,5) )
//              {
//                  LDS_READ_A1B1(3)
//                  Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
//              }
//             if(GetBit(last_k_res_mask_bit,4) && !GetBit(last_k_res_mask_bit,5) )
//              {
//                 LDS_READ_A1B1(3)
//                 Dot2_F32_F16_CLear_A_HI(&A,&B,&C0,&C1,&C2,&C3);
//              }
//               //inter3
//             if(GetBit(last_k_res_mask_bit,6) && GetBit(last_k_res_mask_bit,7)  )
//              {
//              LDS_READ_AB(4)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//             if(GetBit(last_k_res_mask_bit,6) && !GetBit(last_k_res_mask_bit,7)  )
//              {
//                LDS_READ_AB(4)
//                Dot2_F32_F16_CLear_A_HI(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//             //inter4
//             if(GetBit(last_k_res_mask_bit,8) && GetBit(last_k_res_mask_bit,9) )
//              {
//              LDS_READ_A1B1(5)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,8) && !GetBit(last_k_res_mask_bit,9) )
//              {
//                 LDS_READ_A1B1(5)
//                 Dot2_F32_F16_CLear_A_HI(&A,&B,&C0,&C1,&C2,&C3);
//              }
//              //inter5
//              if(GetBit(last_k_res_mask_bit,10) && GetBit(last_k_res_mask_bit,11))
//              {
//              LDS_READ_AB(6)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,10) && !GetBit(last_k_res_mask_bit,11))
//              {
//                LDS_READ_AB(6)
//                Dot2_F32_F16_CLear_A_HI(&A1,&B1,&C0,&C1,&C2,&C3);
//              }      
//             //inter6
//              if(GetBit(last_k_res_mask_bit,12) && GetBit(last_k_res_mask_bit,13))
//              {
//              LDS_READ_A1B1(7)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,12) && !GetBit(last_k_res_mask_bit,13))
//              {
//                  LDS_READ_A1B1(7)
//                  Dot2_F32_F16_CLear_A_HI(&A,&B,&C0,&C1,&C2,&C3);
//              }
           
//              //inter7
//             if(GetBit(last_k_res_mask_bit,14) && GetBit(last_k_res_mask_bit,15) )
//              {
//              LDS_READ_AB_HALF(0)
//              Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//             if(GetBit(last_k_res_mask_bit,14) && !GetBit(last_k_res_mask_bit,15) )
//              {
//                LDS_READ_AB_HALF(0)
//                (&A1,&B1,&C0,&C1,&C2,&C3);
//              }

//              //inter8
//              if(GetBit(last_k_res_mask_bit,16) && GetBit(last_k_res_mask_bit,17))
//              {
//               LDS_READ_A1B1_HALF(1)
//               Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,16) && !GetBit(last_k_res_mask_bit,17))
//              {
//                LDS_READ_A1B1_HALF(1)
//                Dot2_F32_F16_CLear_A_HI(&A,&B,&C0,&C1,&C2,&C3);
//              }
//             //inter9
//             if(GetBit(last_k_res_mask_bit,18) && GetBit(last_k_res_mask_bit,19) )
//              {
//               LDS_READ_AB_HALF(2)
//               Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//             if(GetBit(last_k_res_mask_bit,18) && !GetBit(last_k_res_mask_bit,19) )
//              {
//                 LDS_READ_AB_HALF(2)
//                 Dot2_F32_F16_CLear_A_HI(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
                                  
//             //inter10
//             if(GetBit(last_k_res_mask_bit,20) && GetBit(last_k_res_mask_bit,21) )
//              {
//              LDS_READ_A1B1_HALF(3)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
//              }
//             if(GetBit(last_k_res_mask_bit,20) && !GetBit(last_k_res_mask_bit,21) )
//              {
//              LDS_READ_A1B1_HALF(3)
//              Dot2_F32_F16_CLear_A_HI(&A,&B,&C0,&C1,&C2,&C3);
//              }
//              //inter11
//              if(GetBit(last_k_res_mask_bit,22) && GetBit(last_k_res_mask_bit,23))
//              {
//               LDS_READ_AB_HALF(4)
//               Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,22) && !GetBit(last_k_res_mask_bit,23))
//              {
//               LDS_READ_AB_HALF(4)
//               Dot2_F32_F16_CLear_A_HI(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//              //inter12
//              if(GetBit(last_k_res_mask_bit,24) && GetBit(last_k_res_mask_bit,24))
//              {
//              LDS_READ_A1B1_HALF(5)
//              Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,24) && !GetBit(last_k_res_mask_bit,24))
//              {
//              LDS_READ_A1B1_HALF(5)
//              Dot2_F32_F16_CLear_A_HI(&A,&B,&C0,&C1,&C2,&C3);
//              }
//            //inter13
//              if(GetBit(last_k_res_mask_bit,26) && GetBit(last_k_res_mask_bit,27) )
//              {
//               LDS_READ_AB_HALF(6)
//               Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,26) && !GetBit(last_k_res_mask_bit,27) )
//              {
//               LDS_READ_AB_HALF(6)
//               Dot2_F32_F16_CLear_A_HI(&A1,&B1,&C0,&C1,&C2,&C3);
//              }

//             //inter14
//              if(GetBit(last_k_res_mask_bit,28) && GetBit(last_k_res_mask_bit,29) )
//              {
//                LDS_READ_A1B1_HALF(7)
//                Dot2_F32_F16(&A,&B,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,28) && !GetBit(last_k_res_mask_bit,29) )
//              {
//                LDS_READ_A1B1_HALF(7)
//                Dot2_F32_F16_CLear_A_HI(&A,&B,&C0,&C1,&C2,&C3);
//              }
//             //inter15
//              if(GetBit(last_k_res_mask_bit,30) && GetBit(last_k_res_mask_bit,31))
//              {
//                //LDS_READ_A1B1(15)
//                Dot2_F32_F16(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
//              if(GetBit(last_k_res_mask_bit,30) && !GetBit(last_k_res_mask_bit,31))
//              {
//                //LDS_READ_A1B1(15)
//                Dot2_F32_F16_CLear_A_HI(&A1,&B1,&C0,&C1,&C2,&C3);
//              }
             
//            oc = g *group_out_channels + (64 * blockIdx.x)+ (threadIdx.x) * 4;                          
//            tem =  (blockIdx.z/group) * strideC_0 + oc * strideC_1 + j_res;
//            unsigned int oc_edge = (g + 1)*group_out_channels < strideC_3 ?(g + 1)*group_out_channels : strideC_3;

//            if((oc + 3 < oc_edge) && (j_res + 3 < strideC_1))
//            {
//                   offset0_tmp = tem;
//                   offset1_tmp = (tem + strideC_1);
//                   offset2_tmp = (tem + 2*strideC_1);
//                   offset3_tmp = (tem + 3*strideC_1);

//                   V_cvt_pkrtz_f16_f32(&D,&C0);
//                   *(float2 *)&gemmC[offset0_tmp] =   *(float2 *)&D;
//                   V_cvt_pkrtz_f16_f32(&D,&C1);
//                   *(float2 *)&gemmC[offset1_tmp] =   *(float2 *)&D;
//                   V_cvt_pkrtz_f16_f32(&D,&C2);
//                   *(float2 *)&gemmC[offset2_tmp] =   *(float2 *)&D;
//                   V_cvt_pkrtz_f16_f32(&D,&C3);
//                   *(float2 *)&gemmC[offset3_tmp] =   *(float2 *)&D;
             
//            }
//            else
//            {
//                 V_cvt_pkrtz_f16_f32(&D,&C0);
//                 offset0_tmp = (oc < strideC_3 && (j_res + 0) < strideC_1)? (tem):-1;
//                 offset1_tmp = (oc < strideC_3 && (j_res + 1) < strideC_1)? ((tem + 1)):-1;
//                 offset2_tmp = (oc < strideC_3 && (j_res + 2) < strideC_1)? ((tem + 2)):-1;
//                 offset3_tmp = (oc < strideC_3 && (j_res + 3) < strideC_1)? ((tem + 3)):-1;

//                 bufferstoreC(&D,&offset0_tmp,&offset1_tmp,&offset2_tmp,&offset3_tmp,&globalStoreC);

//                 V_cvt_pkrtz_f16_f32(&D,&C1);
//                 tem +=  strideC_1;
//                 offset0_tmp = (oc + 1< strideC_3 && (j_res + 0) < strideC_1)? (tem):-1;
//                 offset1_tmp = (oc + 1< strideC_3 && (j_res + 1) < strideC_1)? ((tem + 1)):-1;
//                 offset2_tmp = (oc + 1< strideC_3 && (j_res + 2) < strideC_1)? ((tem + 2)):-1;
//                 offset3_tmp = (oc + 1< strideC_3 && (j_res + 3) < strideC_1)? ((tem + 3)):-1;
//                 bufferstoreC(&D,&offset0_tmp,&offset1_tmp,&offset2_tmp,&offset3_tmp,&globalStoreC);

//                 V_cvt_pkrtz_f16_f32(&D,&C2);
//                 tem +=  strideC_1;
//                 offset0_tmp = (oc + 2< strideC_3 && (j_res + 0) < strideC_1)? (tem):-1;
//                 offset1_tmp = (oc + 2< strideC_3 && (j_res + 1) < strideC_1)? ((tem + 1)):-1;
//                 offset2_tmp = (oc + 2< strideC_3 && (j_res + 2) < strideC_1)? ((tem + 2)):-1;
//                 offset3_tmp = (oc + 2< strideC_3 && (j_res + 3) < strideC_1)? ((tem + 3)):-1;
//                 bufferstoreC(&D,&offset0_tmp,&offset1_tmp,&offset2_tmp,&offset3_tmp,&globalStoreC);

//                 V_cvt_pkrtz_f16_f32(&D,&C3);
//                 tem +=  strideC_1;
//                 offset0_tmp = (oc + 3< strideC_3 && (j_res + 0) < strideC_1)? (tem):-1;
//                 offset1_tmp = (oc + 3< strideC_3 && (j_res + 1) < strideC_1)? ((tem + 1)):-1;
//                 offset2_tmp = (oc + 3< strideC_3 && (j_res + 2) < strideC_1)? ((tem + 2)):-1;
//                 offset3_tmp = (oc + 3< strideC_3 && (j_res + 3) < strideC_1)? ((tem + 3)):-1;
//                 bufferstoreC(&D,&offset0_tmp,&offset1_tmp,&offset2_tmp,&offset3_tmp,&globalStoreC);
                    
               
//            }

// }

// typedef unsigned short half;
// half float_to_half(float m)
// {
//     unsigned long m2 = *(unsigned long*)(&m);    
//     // 强制把float转为unsigned long
//     // 截取后23位尾数，右移13位，剩余10位；符号位直接右移16位；
//     // 指数位麻烦一些，截取指数的8位先右移13位(左边多出3位不管了)
//     // 之前是0~255表示-127~128, 调整之后变成0~31表示-15~16
//     // 因此要减去127-15=112(在左移10位的位置).
//     unsigned short t = ((m2 & 0x007fffff) >> 13) | ((m2 & 0x80000000) >> 16) 
//         | (((m2 & 0x7f800000) >> 13) - (112 << 10));           
//     if(m2 & 0x1000) 
//         t++;                   // 四舍五入(尾数被截掉部分的最高位为1, 则尾数剩余部分+1)
//     half h = *(half*)(&t);     // 强制转为half
//     return h ;
// }
// float half_to_float(half n)
// {
//     unsigned short frac = (n & 0x3ff) | 0x400;
//     int exp = ((n & 0x7c00) >> 10) - 25;
//     float m;
//     if(frac == 0 && exp == 0x1f)
//         m = INFINITY;
//     else if (frac || exp)
//         m = frac * pow(2, exp);
//     else
//         m = 0;
//     return (n & 0x8000) ? -m : m;
// }

// static float half_to_float(_Float16 hf)
// {
//     FP16 h = *((FP16*)&hf);

//     static const FP32 magic = { 113 << 23 };
//     static const uint shifted_exp = 0x7c00 << 13; // exponent mask after shift
//     FP32 o;

//     o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
//     uint exp = shifted_exp & o.u;   // just the exponent
//     o.u += (127 - 15) << 23;        // exponent adjust

//     // handle exponent special cases
//     if (exp == shifted_exp) // Inf/NaN?
//         o.u += (128 - 16) << 23;    // extra exp adjust
//     else if (exp == 0) // Zero/Denormal?
//     {
//         o.u += 1 << 23;             // extra exp adjust
//         o.f -= magic.f;             // renormalize
//     }

//     o.u |= (h.u & 0x8000) << 16;    // sign bit
//     return o.f;
// }

// static float half_to_float_1(_Float16 _x)
// {
// 	unsigned short n = *((unsigned short*)&_x);
// 	unsigned int x = (unsigned int)n;
// 	x = x & 0xffff;
// 	unsigned int sign = x &0x8000;
// 	unsigned int expoent_f16 = (x&0x7c00)>>10;
// 	unsigned int mantissa_f16 = x&0x03ff;
// 	unsigned int y = sign << 16;
// 	unsigned int expoent_f32;
// 	unsigned int mantissa_f32;
// 	unsigned int first_1_pos = 0;
// 	unsigned int mask ;
// 	unsigned int hx;
	
// 	printf("%d,%d\n",expoent_f16,mantissa_f16);
	
// 	hx = x&0x7fff;
	
// 	if(hx==0)
// 		return *((float*)&y);
	
// 	if(hx==0x7c00)
// 	{
// 		y |= 0x7f800000;
// 		return *((float*)&y);
// 	}
	
// 	if(hx>0x7c00)
// 	{
// 		y = 0x7fc00000;
// 		return *((float*)&y);
// 	}
	
// 	expoent_f32 = 0x70 + expoent_f16;
// 	mantissa_f32 = mantissa_f16<<13;
	
// 	for(first_1_pos=0;first_1_pos<10;first_1_pos++)
// 	{
// 		if((mantissa_f16>>(first_1_pos+1)) == 0)
// 			break;
// 	}
	
// 	if(expoent_f16==0)
// 	{
// 		mask = (1<<23)-1;
// 		expoent_f32 = expoent_f32 - (10-first_1_pos) + 1;
// 		mantissa_f32 = mantissa_f32 << (10-first_1_pos);
// 		mantissa_f32 = mantissa_f32 & mask;
// 	}
	
// 	y = y | (expoent_f32<<23) | mantissa_f32;
	
// 	return *((float*)&y);
		
// }

void convolution_2d_fp16(hipStream_t stream, 
                const argument& result, 
                const argument& x,
                const argument& w,
                const std::vector<std::size_t> padding,
                const std::vector<std::size_t> stride,
                const std::vector<std::size_t> dilation,
                const int group)
{
    
    // 参数
    // unsigned int batch = x.get_shape().lens()[0];
    // unsigned int channels = x.get_shape().lens()[1];
    // unsigned int num_kernel = w.get_shape().lens()[0];
	// unsigned int height = x.get_shape().lens()[2];
    // unsigned int width = x.get_shape().lens()[3];
    // unsigned int pad_h = padding[0];
	// unsigned int pad_w = padding[1];
    // unsigned int r = w.get_shape().lens()[2]; // 卷积核大小
    // unsigned int s = w.get_shape().lens()[3];
	// unsigned int stride_h =stride[0];
	// unsigned int stride_w =stride[1];
    // unsigned int dilate_h =dilation[0];
	// unsigned int dilate_w =dilation[1];

    // int dilate_filter_h = dilate_h * (r - 1) + 1;
	// int dilate_filter_w = dilate_w * (s - 1) + 1;

	// int o_w = (width  + 2*pad_w - dilate_filter_w)/stride_w  + 1;
	// int o_h = (height + 2*pad_h - dilate_filter_h)/stride_h  + 1 ;

    // int M = num_kernel/group;
    // int N = o_h*o_w;
    // int K = (channels/group)*r*s;
    // unsigned int numgroupM = ((M + THREADS_NUM_M*T_M - 1)/(THREADS_NUM_M*T_M));
    // unsigned int numgroupN = (N + THREADS_NUM_N*T_N -1)/(THREADS_NUM_N*T_N);
    // unsigned int numgroupZ = group *batch ;

    // int last_k_res_mask_bit = 0;
	// int last_k_res =  (K%depthU_C);

	// for(int i = 0;i < last_k_res;i++)
	// {
    //    SetBit(last_k_res_mask_bit,i);
	// }
    
      
    // unsigned int strideA_0 = (channels/group)*r*s;
    // unsigned int strideA_1 = r*s;
    // unsigned int strideA_2 = s;
    // unsigned int strideA_3 = o_h;//为了复用这个寄存器
    // unsigned int strideA_4 = num_kernel*(channels/group)*r*s;
    // unsigned int strideA_5 = channels/group;
    // unsigned int strideB_0 = (channels/group)*width*height;
    // unsigned int strideB_1 = width*height;
    // unsigned int strideB_2 = width;
    // unsigned int strideB_3 = height;//为了复用这个寄存器
    // unsigned int strideB_4 = batch*channels*width*height;
    // unsigned int strideC_0 = num_kernel*o_w*o_h;
    // unsigned int strideC_1 = o_w*o_h;
    // unsigned int strideC_2 = o_w;
    // unsigned int strideC_3 = num_kernel;//为了复用这个寄存器
    // unsigned int remdinergroupX = M%(THREADS_NUM_M*T_M);
    // unsigned int remdinergroupY = N%(THREADS_NUM_N*T_N);

    // unsigned int group_out_channels = num_kernel/group;
    // unsigned int group_block_nums = (group_out_channels - 1)/64 + 1;


    // dim3 threads(THREADS_NUM_M,THREADS_NUM_N,1);
    // dim3 groups(numgroupM, numgroupN, numgroupZ);

    // hipLaunchKernelGGL(implicit_gemmComAlg,
    //                                         groups,
    //                                         threads,
    //                                         0,
    //                                         stream,
    //                                         (_Float16 *)w.data(), 
    //                                         (_Float16 *)x.data(), 
    //                                         (_Float16 *)result.data(), 
    //                                         strideA_0,
    //                                         strideA_1,
    //                                         strideA_2,
    //                                         strideA_3,
    //                                         strideA_4,
    //                                         strideB_0,
    //                                         strideB_1,
    //                                         strideB_2,
    //                                         strideB_3,
    //                                         strideB_4,
    //                                         strideC_0,
    //                                         strideC_1,
    //                                         strideC_2,
    //                                         strideC_3,
    //                                         last_k_res_mask_bit,     
    //                                         last_k_res,     
    //                                         stride_h,  
    //                                         stride_w,
    //                                         pad_h,
    //                                         pad_w,
    //                                         dilate_h,
    //                                         dilate_w,
    //                                         group,
    //                                         group_out_channels,
    //                                         group_block_nums,
    //                                         numgroupM,
    //                                         numgroupN,
    //                                         remdinergroupX,
    //                                         remdinergroupY);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
