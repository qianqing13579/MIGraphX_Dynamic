// 将汇编修改为hip实现后，只能适用于pad=0的情况

#include <migraphx/gpu/device/convolution_2d_fp32.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/SimpleLog.h>
#include <migraphx/gpu/gemm_impl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

typedef long BB __attribute__((ext_vector_type(2)));

#define depthU_C 16
#define T_M 4
#define T_M_bit 2
#define T_N 4
#define T_N_bit 2
#define THREADS_NUM_M 16
#define THREADS_NUM_N 16
#define READ_A_Num 4
#define READ_A_ROW_bit 2
#define READ_A_COL 3

#define READ_B_Num 4
#define READ_B_COL_bit 2
#define READ_B_COL 3

#define GetBit(x,y) ((x) >> (y)&1)	//获取将x的第y位（0或1）
#define SetBit(x,y) x|=(1<<y) 		//将X的第Y位置1
#define ClrBit(x,y) x&=~(1<<y) 		//将X的第Y位清0



#define LDS_SIZE 2048
#define LDS_OFFSET    1024



#define MAC_4X4_F32 C00 += A.x*B.x;\
                    C01 += A.x*B.y;\
                    C02 += A.x*B.z;\
                    C03 += A.x*B.w;\
                    C10 += A.y*B.x;\
                    C11 += A.y*B.y;\
                    C12 += A.y*B.z;\
                    C13 += A.y*B.w;\
                    C20 += A.z*B.x;\
                    C21 += A.z*B.y;\
                    C22 += A.z*B.z;\
                    C23 += A.z*B.w;\
                    C30 += A.w*B.x;\
                    C31 += A.w*B.y;\
                    C32 += A.w*B.z;\
                    C33 += A.w*B.w;\

#define MAC_4X4_F32_1  \
C00 += A1.x*B1.x;\
C01 += A1.x*B1.y;\
C02 += A1.x*B1.z;\
C03 += A1.x*B1.w;\
C10 += A1.y*B1.x;\
C11 += A1.y*B1.y;\
C12 += A1.y*B1.z;\
C13 += A1.y*B1.w;\
C20 += A1.z*B1.x;\
C21 += A1.z*B1.y;\
C22 += A1.z*B1.z;\
C23 += A1.z*B1.w;\
C30 += A1.w*B1.x;\
C31 += A1.w*B1.y;\
C32 += A1.w*B1.z;\
C33 += A1.w*B1.w;\



// #define LDS_READ_AB(times) \
// A = *((float4 *)(&ldsa[lra+ times*64+0])); \
// B = *((float4 *)(&ldsb[lrb+ (times>>1)*32+(times&1)*4])); \

// #define LDS_READ_A1B1(times) \
// A1 = *((float4 *)(&ldsa[lra+ times*64+0])); \
// B1 = *((float4 *)(&ldsb[lrb+ (times>>1)*32+(times&1)*4])); \

 //4x4 4x4 性能比4x4 8x2性能好
#define LDS_READ_AB(times) \
B = *((float4 *)(&ldsb[lrb+ (times)*4])); \
A = *((float4 *)(&ldsa[lra+ times*64+0])); \


#define LDS_READ_A1B1(times) \
B1 = *((float4 *)(&ldsb[lrb+ (times)*4])); \
A1 = *((float4 *)(&ldsa[lra+ times*64+0])); \


__attribute__((device)) static inline void bufferloadA(float4 *gA, int* offsetA0,int* offsetA1, int* offsetA2,int* offsetA3,BB* globalReadA)
{
    asm volatile( 
                                    

                "buffer_load_dword %0,%4,%8,0, offen,offset:0\n"
                "buffer_load_dword %1,%5,%8,0, offen offset:0\n"               
                "buffer_load_dword %2,%6,%8,0, offen offset:0\n"                
                "buffer_load_dword %3,%7,%8,0, offen offset:0\n"

                :"=v"(gA->x),"=v"(gA->y),"=v"(gA->z),"=v"(gA->w),"+v"(*offsetA0), "+v"(*offsetA1),"+v"(*offsetA2),"+v"(*offsetA3),"+s"(*globalReadA));
}
 __attribute__((device)) static inline void bufferloadB(float4 *gB, int* offsetA0,int* offsetA1, int* offsetA2,int* offsetA3,BB* globalReadB)
{
    asm volatile( 

                "buffer_load_dword %0,%4,%8,0,idxen \n"
                "buffer_load_dword %1,%5,%8,0,idxen \n"               
                "buffer_load_dword %2,%6,%8,0,idxen \n"              
                "buffer_load_dword %3,%7,%8,0,idxen \n"

                 
                :"=v"(gB->x),"=v"(gB->y),"=v"(gB->z),"=v"(gB->w),"+v"(*offsetA0), "+v"(*offsetA1),"+v"(*offsetA2),"+v"(*offsetA3),"+s"(*globalReadB));
} 
__attribute__((device)) static inline void bufferstoreC(float *C0,float *C1,float *C2,float *C3, int* offsetA0,int* offsetA1, int* offsetA2,int* offsetA3,BB* globalStoreC)
{
    asm volatile( 
                                    
                "buffer_store_dword %0,%4,%8,0, idxen\n"
                "buffer_store_dword %1,%5,%8,0, idxen\n"                 
                "buffer_store_dword %2,%6,%8,0, idxen\n"                 
                "buffer_store_dword %3,%7,%8,0, idxen\n"
                    
                "s_waitcnt vmcnt(0)\n\t"
                :"+v"(*C0),"+v"(*C1),"+v"(*C2),"+v"(*C3),"+v"(*offsetA0), "+v"(*offsetA1),"+v"(*offsetA2),"+v"(*offsetA3),"+s"(*globalStoreC));
} 

 __attribute__((device)) static inline void doublebuffer(unsigned *wa,unsigned *wb, unsigned *ra,unsigned *rb)
{
    asm volatile( 
                                
                   "v_add_u32_e32 v45, 0x400,v45 \n"
                   "v_and_b32_e32 v45, 0x7ff,v45 \n"
                   "v_add_u32_e32 v46, 0x400,v46 \n"
                   "v_and_b32_e32 v46, 0x7ff,v46 \n"
                   "v_add_u32_e32 v55, 0x400,v55 \n"
                   "v_and_b32_e32 v55, 0x7ff,v55 \n"
                   "v_add_u32_e32 %3, 0x400,%3 \n" 
                   "v_and_b32_e64 %3, 0x7ff,%3 \n"              
                  :"+v"(wa),"+v"(wb),"+v"(ra),"+v"(rb));

}

//加汇编-1读取
__global__ void ImplicitGemmCommDilateGroupTA(float* gemmA,
                                             float* gemmB,
                                             float* gemmC,
                                             unsigned int strideA_0,
                                             unsigned int strideA_1,
                                             unsigned int strideA_2,
                                             unsigned int strideA_3,
                                             unsigned int strideA_4,
                                             unsigned int strideB_0,
                                             unsigned int strideB_1,
                                             unsigned int strideB_2,
                                             unsigned int strideB_3,
                                             unsigned int strideB_4,
                                             unsigned int strideC_0,
                                             unsigned int strideC_1,
                                             unsigned int strideC_2,
                                             unsigned int strideC_3,
                                             unsigned int strideC_4,
                                             unsigned int last_k_res_mask_bit,
                                             unsigned int last_k_res,
                                             unsigned int stride_H,
                                             unsigned int stride_W,
                                             unsigned int pad_H,
                                             unsigned int pad_W,
                                             unsigned int dilate_H,
                                             unsigned int dilate_W,
                                             unsigned int group,
                                             unsigned int group_out_channels,
                                             unsigned int group_block_nums,
                                             unsigned int numgroupX,
                                             unsigned int numgroupY,
                                             unsigned int remdinergroupX,
                                             unsigned int remdinergroupY)
{

         // bufferload的缓冲配置
         BB globalReadA;
         globalReadA.x = (long)gemmA;
         globalReadA.x = (globalReadA.x | (((long)(0x4 << 16))<<32)); //const_stride    
         globalReadA.y = (((((long)0x20000)<<32) | 0xFFFFFFFE));
         BB globalReadB;
         globalReadB.x = (long)gemmB;
         globalReadB.x = (globalReadB.x | (((long)(0x4 << 16))<<32)); //const_stride      
         globalReadB.y = (((((long)0x20000)<<32) | 0xFFFFFFFE));
         BB globalStoreC;
         globalStoreC.x = (long)gemmC;
         globalStoreC.x = (globalStoreC.x | (((long)(0x4 << 16))<<32)); //const_stride   
         globalStoreC.y = (((((long)0x20000)<<32) | 0xFFFFFFFE));

         BB globalReadA1 = globalReadA;
         BB globalReadB1 = globalReadB;
         BB globalStoreC1 = globalStoreC;

        int threadidInblock =  threadIdx.x + threadIdx.y*blockDim.x;

        int readcol = 0;//threadidInblock & READ_A_COL;
        int readrow = 0;// threadidInblock >> READ_A_ROW_bit ; 

        // readcol = threadidInblock & READ_A_COL;
        // readrow = threadidInblock >> READ_A_ROW_bit ;

        if(strideA_0 <= 1024)
        {
           readcol = threadidInblock & READ_A_COL;
           readrow = threadidInblock >> READ_A_ROW_bit ;
        }
        else if(strideA_0 > 1024 && strideA_0 <= 4096)
        {
           readcol = threadidInblock >> 6;
           readrow = threadidInblock & 63;
        }
        else
        {
           readcol = (threadIdx.y >>3)*2 +  (threadIdx.x >>3);
           readrow = (threadIdx.x & 7) + (threadIdx.y & 7) * 8;
        }
    
        //GEMM
         __shared__ float ldsa[LDS_SIZE];
         __shared__ float ldsb[LDS_SIZE];

         unsigned int i = 0; 
         float4 gA;
         float4 gB;

         float4 A;
         float4 B;
         float4 A1;
         float4 B1;
 
        //控制边界
        //Gemm 到Conv 坐标转换
        //dst_c
        unsigned int g = blockIdx.z % group;
        unsigned int oc = g *group_out_channels + (64 *  blockIdx.x) + readrow ;  
        unsigned int n =  (blockIdx.z/group) * strideB_0*group  + g * strideB_0;
        unsigned int j_res = 64 * blockIdx.y  + (threadIdx.y) * 4;
        //src_b 
        int ReadAFlag = 0;
        int ReadBFlag = 0;
        int oh0;
        int ow0;
        int oh1;
        int ow1;
        int oh2;
        int ow2;
        int oh3;
        int ow3;
        //src_a
        unsigned int ic = (threadIdx.x) /(strideA_3);
        unsigned int k_res =(threadIdx.x) %(strideA_3);
        unsigned int fh  = (k_res / strideA_2)* dilate_H;
        unsigned int fw  = (k_res % strideA_2)* dilate_W;
        //step_k
        //unsigned int step_k = 0;
        unsigned int tem = 0;
        unsigned int tem1 = 0;

        int offset0_tmp;
        int offset1_tmp;
        int offset2_tmp;
        int offset3_tmp;

  
        // gA.x = 0.0;
        // gA.y = 0.0;
        // gA.z = 0.0;
        // gA.w = 0.0;
        // gB.x = 0.0;
        // gB.y = 0.0;
        // gB.z = 0.0;
        // gB.w = 0.0;
         //global->lds    
        //A     
         unsigned int A_edge = strideA_4 < (g + 1)*group_out_channels * strideA_0 ? strideA_4 : (g + 1)*group_out_channels * strideA_0;
         int base_A = (oc * strideA_0  + 0 + readcol*READ_A_Num);
         offset0_tmp =(base_A < A_edge  )?((0 + base_A)):-1;
         asm volatile(
                "buffer_load_dwordx4 %0,%1,%2,0, idxen \n"
            :"=v"(gA), "+v"(offset0_tmp), "+s"(globalReadA1));
        
        //B
         oh0 = ((j_res + 0) / strideC_2) * stride_H - pad_H;
         ow0 = ((j_res + 0) % strideC_2) * stride_W - pad_W;

         oh1 = ((j_res + 1) / strideC_2) * stride_H - pad_H;
         ow1 = ((j_res + 1) % strideC_2) * stride_W - pad_W;

         oh2 = ((j_res + 2) / strideC_2) * stride_H - pad_H;
         ow2 = ((j_res + 2) % strideC_2) * stride_W - pad_W;

         oh3 = ((j_res + 3) / strideC_2) * stride_H - pad_H;
         ow3 = ((j_res + 3) % strideC_2) * stride_W - pad_W;

       

        tem = (n  + ic * strideB_1 + fh * strideB_2 + fw) ; 

        // int offset0 = 4*(oh0 *strideB_2 +  ow0);       
        // int offset1 = 4*(oh1 *strideB_2 +  ow1);
        // int offset2 = 4*(oh2 *strideB_2 +  ow2);
        // int offset3 = 4*(oh3 *strideB_2 +  ow3);


        offset0_tmp =(((fh  + oh0) < strideB_3 && fw  + ow0 < strideB_2)&& (threadIdx.x < strideA_0))?(tem + (oh0 *strideB_2 +  ow0)):-1;
        offset1_tmp =(((fh  + oh1) < strideB_3 && fw  + ow1 < strideB_2)&& (threadIdx.x < strideA_0))?(tem + (oh1 *strideB_2 +  ow1)):-1;
        offset2_tmp =(((fh  + oh2) < strideB_3 && fw  + ow2 < strideB_2)&& (threadIdx.x < strideA_0))?(tem + (oh2 *strideB_2 +  ow2)):-1;
        offset3_tmp =(((fh  + oh3) < strideB_3 && fw  + ow3 < strideB_2)&& (threadIdx.x < strideA_0))?(tem + (oh3 *strideB_2 +  ow3)):-1;
        
 
        // bufferloadB(&gB, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadB1);
        if(offset0_tmp!=-1)
        {
            gB.x =     gemmB[offset0_tmp];
        }
        if(offset1_tmp!=-1)
        {
            gB.y =     gemmB[offset1_tmp];
        }
        if(offset2_tmp!=-1)
        {
            gB.z =     gemmB[offset2_tmp];
        }
        if(offset3_tmp!=-1)
        {
            gB.w =     gemmB[offset3_tmp];
        }



         asm volatile("s_waitcnt vmcnt(0)\n\t");
         //__syncthreads();
         //一个线程读取4个数据一个块256个线程一次读取1024个数据 A B总共2048个数据
        //  unsigned int lwa = readrow + readcol * READ_A_Num * 64;
        //  unsigned int lwb = (threadIdx.y >>2)*256  + (threadIdx.y&3) * 8 + (threadIdx.x >>1)*32 +(threadIdx.x &1)*4;
 
        //  unsigned int lra = 4 * threadIdx.x ;
        //  unsigned int lrb = (threadIdx.y >>2)*256  + (threadIdx.y&3) * 8;

        // // 4x4 4x4 性能比4x4 8x2性能好
        // unsigned int lwa = readrow + readcol * READ_A_Num * 64;
        // unsigned int lwb = (threadIdx.y >>2)*256  + (threadIdx.y&3) * 16 + (threadIdx.x >>2)*64 +(threadIdx.x & 3)*4;

        // unsigned int lra = 4 * threadIdx.x ;
        // unsigned int lrb = (threadIdx.y >>2)*256  + (threadIdx.y&3) * 16;

       // B 4x4 4x4 性能比16x1 1x16性能好
        unsigned int lwa = readrow + readcol * READ_A_Num * 64;
        unsigned int lwb = (threadIdx.y)*64  + (threadIdx.x)*4 ;

        unsigned int lra = 4 * threadIdx.x ;
        unsigned int lrb = (threadIdx.y)*64 ;



        //write lds
        ldsa[lwa + 0*64] = gA.x;
        ldsa[lwa + 1*64] = gA.y;
        ldsa[lwa + 2*64] = gA.z;
        ldsa[lwa + 3*64] = gA.w;
        *((float4 *)&ldsb[lwb]) = gB;

        // lwa = (lwa + LDS_OFFSET) & 2047;
        // lwb = (lwb + LDS_OFFSET) & 2047;

        __syncthreads();
        //其他计算
         float C00 = 0.0;
         float C01 = 0.0;
         float C02 = 0.0;
         float C03 = 0.0;
         float C10 = 0.0;
         float C11 = 0.0;
         float C12 = 0.0;
         float C13 = 0.0;
         float C20 = 0.0;
         float C21 = 0.0;
         float C22 = 0.0;
         float C23 = 0.0;
         float C30 = 0.0;
         float C31 = 0.0;
         float C32 = 0.0; 
         float C33 = 0.0;  


        //preload
         LDS_READ_AB(0)
         tem1 = threadIdx.x + 16;
         
         for( i = 0;i < strideA_0 - last_k_res;i+=16)
         {
              //global->lds
              //read_row_flag = 0;
            ic = (i + tem1) / strideA_3;
            k_res=(i + tem1) % strideA_3;
            fh = (k_res / strideA_2)*dilate_H;
            fw = (k_res % strideA_2)*dilate_W;

            tem = (n  + ic * strideB_1 + fh * strideB_2 + fw) ;   

            offset0_tmp =(((fh  + oh0) < strideB_3 && fw  + ow0 < strideB_2 ) && i + tem1  < strideA_0)?(tem + (oh0 *strideB_2 +  ow0)):-1;
            offset1_tmp =(((fh  + oh1) < strideB_3 && fw  + ow1 < strideB_2 ) && i + tem1  < strideA_0)?(tem + (oh1 *strideB_2 +  ow1)):-1;
            offset2_tmp =(((fh  + oh2) < strideB_3 && fw  + ow2 < strideB_2 ) && i + tem1  < strideA_0)?(tem + (oh2 *strideB_2 +  ow2)):-1;
            offset3_tmp =(((fh  + oh3) < strideB_3 && fw  + ow3 < strideB_2 ) && i + tem1  < strideA_0)?(tem + (oh3 *strideB_2 +  ow3)):-1;
                                            
            // bufferloadB(&gB, &offset0_tmp,&offset1_tmp, &offset2_tmp,&offset3_tmp,&globalReadB);
            if(offset0_tmp!=-1)
            {
                gB.x =     gemmB[offset0_tmp];
            }
            if(offset1_tmp!=-1)
            {
                gB.y =     gemmB[offset1_tmp];
            }
            if(offset2_tmp!=-1)
            {
                gB.z =     gemmB[offset2_tmp];
            }
            if(offset3_tmp!=-1)
            {
                gB.w =     gemmB[offset3_tmp];
            }

             //preload 
             //inter0
             LDS_READ_A1B1(1)
             MAC_4X4_F32

             //inter1
             LDS_READ_AB(2)
             MAC_4X4_F32_1
       
             //inter2
             LDS_READ_A1B1(3)
             MAC_4X4_F32

             //inter3
             LDS_READ_AB(4)
             MAC_4X4_F32_1

             //inter4
             LDS_READ_A1B1(5)
             MAC_4X4_F32

             //inter5
             LDS_READ_AB(6)
             MAC_4X4_F32_1

            //inter6
             LDS_READ_A1B1(7)
             MAC_4X4_F32

             //inter7
             LDS_READ_AB(8)
             MAC_4X4_F32_1


            //inter8
             LDS_READ_A1B1(9)
             MAC_4X4_F32

             //inter9
             LDS_READ_AB(10)
             MAC_4X4_F32_1

            //inter10
             LDS_READ_A1B1(11)
             MAC_4X4_F32

             //inter11
             LDS_READ_AB(12)
             MAC_4X4_F32_1

             //inter12
             LDS_READ_A1B1(13)
             MAC_4X4_F32

             //inter13
             LDS_READ_AB(14)
             MAC_4X4_F32_1

             //inter14
             LDS_READ_A1B1(15)
             MAC_4X4_F32
             //write lds    


            tem = (base_A + i + 16);
            offset0_tmp =((0 + tem) < A_edge )?((0 + tem)):-1;
            asm volatile(
                "buffer_load_dwordx4 %0,%1,%2,0, idxen \n"
            :"=v"(gA), "+v"(offset0_tmp), "+s"(globalReadA));

            asm volatile("s_waitcnt vmcnt(0)\n\t");

            __syncthreads();
  
            ldsa[lwa + 0*64] = gA.x;
            ldsa[lwa + 1*64] = gA.y;
            ldsa[lwa + 2*64] = gA.z;
            ldsa[lwa + 3*64] = gA.w;
            *((float4 *)&ldsb[lwb]) = gB;

            // lwa = (lwa + LDS_OFFSET) & 2047;
            // lwb = (lwb + LDS_OFFSET) & 2047;
            // lra = (lra + LDS_OFFSET) & 2047;
            // lrb = (lrb + LDS_OFFSET) & 2047;
       
             __syncthreads();
            //inter15 //preload

             LDS_READ_AB(0)
             MAC_4X4_F32_1
             
         }
            //last inter
            ////inter0
             if(GetBit(last_k_res_mask_bit,0))
             {
             LDS_READ_A1B1(1)
             MAC_4X4_F32
             }

             //inter1
            if(GetBit(last_k_res_mask_bit,1))
             {
             LDS_READ_AB(2)
             MAC_4X4_F32_1
             }
       
            //inter2
            if(GetBit(last_k_res_mask_bit,2))
             {
             LDS_READ_A1B1(3)
             MAC_4X4_F32
             }
              //inter3
            if(GetBit(last_k_res_mask_bit,3))
             {
             LDS_READ_AB(4)
             MAC_4X4_F32_1
             }

            //inter4
            if(GetBit(last_k_res_mask_bit,4))
             {
             LDS_READ_A1B1(5)
             MAC_4X4_F32
             }

             //inter5
             if(GetBit(last_k_res_mask_bit,5))
             {
             LDS_READ_AB(6)
             MAC_4X4_F32_1
             }

            //inter6
             if(GetBit(last_k_res_mask_bit,6))
             {
             LDS_READ_A1B1(7)
             MAC_4X4_F32
             }

             //inter7
            if(GetBit(last_k_res_mask_bit,7))
             {
             LDS_READ_AB(8)
             MAC_4X4_F32_1
             }

            //inter8
             if(GetBit(last_k_res_mask_bit,8))
             {
             LDS_READ_A1B1(9)
             MAC_4X4_F32
             }
            //inter9
            if(GetBit(last_k_res_mask_bit,9))
             {
             LDS_READ_AB(10)
             MAC_4X4_F32_1
             }

            //inter10
            if(GetBit(last_k_res_mask_bit,10))
             {
             LDS_READ_A1B1(11)
             MAC_4X4_F32
             }

             //inter11
             if(GetBit(last_k_res_mask_bit,11))
             {
             LDS_READ_AB(12)
             MAC_4X4_F32_1
             }

             //inter12
             if(GetBit(last_k_res_mask_bit,12))
             {
             LDS_READ_A1B1(13)
             MAC_4X4_F32
             }

             //inter13
             if(GetBit(last_k_res_mask_bit,13))
             {
             LDS_READ_AB(14)
             MAC_4X4_F32_1
             }

            //inter14
             if(GetBit(last_k_res_mask_bit,14))
             {
              //LDS_READ_AB(14)
              MAC_4X4_F32
             }


            //__syncthreads();
                   
          oc = g *group_out_channels + (64 * blockIdx.x)+ (threadIdx.x) * 4;  
          tem =  (blockIdx.z/group) * strideC_0 + oc * strideC_1 + j_res;    
          unsigned int oc_edge = (g + 1)*group_out_channels < strideC_3 ?(g + 1)*group_out_channels : strideC_3;

           offset0_tmp = tem;
           offset1_tmp = tem + strideC_1;
           offset2_tmp = tem + 2*strideC_1;
           offset3_tmp = tem + 3*strideC_1;

           if((oc + 3 < oc_edge) && (j_res + 3 < strideC_1))
           {
                  gemmC[offset0_tmp] = C00;
                  gemmC[offset0_tmp + 1] = C01;
                  gemmC[offset0_tmp + 2] = C02;
                  gemmC[offset0_tmp + 3] = C03;

                  gemmC[offset1_tmp] = C10;
                  gemmC[offset1_tmp + 1] = C11;
                  gemmC[offset1_tmp + 2] = C12;
                  gemmC[offset1_tmp + 3] = C13;

                  gemmC[offset2_tmp] = C20;
                  gemmC[offset2_tmp + 1] = C21;
                  gemmC[offset2_tmp + 2] = C22;
                  gemmC[offset2_tmp + 3] = C23;

                  gemmC[offset3_tmp] = C30;
                  gemmC[offset3_tmp + 1] = C31;
                  gemmC[offset3_tmp + 2] = C32;
                  gemmC[offset3_tmp + 3] = C33;

           }
           else
           {
      

                offset0_tmp = (oc < oc_edge && (j_res) < strideC_1)? (tem):-1;
                offset1_tmp = (oc < oc_edge && (j_res + 1) < strideC_1)? ((tem + 1)):-1;
                offset2_tmp = (oc < oc_edge && (j_res + 2) < strideC_1)? ((tem + 2)):-1;
                offset3_tmp = (oc < oc_edge && (j_res + 3) < strideC_1)? ((tem + 3)):-1;
                bufferstoreC(&C00,&C01,&C02,&C03,&offset0_tmp,&offset1_tmp,&offset2_tmp,&offset3_tmp,&globalStoreC);
            

                tem +=  strideC_1;
                offset0_tmp = (oc + 1< oc_edge && (j_res) < strideC_1)? (tem):-1;
                offset1_tmp = (oc + 1< oc_edge && (j_res + 1) < strideC_1)? ((tem + 1)):-1;
                offset2_tmp = (oc + 1< oc_edge && (j_res + 2) < strideC_1)? ((tem + 2)):-1;
                offset3_tmp = (oc + 1< oc_edge && (j_res + 3) < strideC_1)? ((tem + 3)):-1;
                bufferstoreC(&C10,&C11,&C12,&C13,&offset0_tmp,&offset1_tmp,&offset2_tmp,&offset3_tmp,&globalStoreC);


                tem +=  strideC_1;
                offset0_tmp = (oc + 2< oc_edge && (j_res) < strideC_1)? (tem):-1;
                offset1_tmp = (oc + 2< oc_edge && (j_res + 1) < strideC_1)? ((tem + 1)):-1;
                offset2_tmp = (oc + 2< oc_edge && (j_res + 2) < strideC_1)? ((tem + 2)):-1;
                offset3_tmp = (oc + 2< oc_edge && (j_res + 3) < strideC_1)? ((tem + 3)):-1;
                bufferstoreC(&C20,&C21,&C22,&C23,&offset0_tmp,&offset1_tmp,&offset2_tmp,&offset3_tmp,&globalStoreC);


                tem +=  strideC_1;
                offset0_tmp = (oc + 3< oc_edge && (j_res) < strideC_1)? (tem):-1;
                offset1_tmp = (oc + 3< oc_edge && (j_res + 1) < strideC_1)? ((tem + 1)):-1;
                offset2_tmp = (oc + 3< oc_edge && (j_res + 2) < strideC_1)? ((tem + 2)):-1;
                offset3_tmp = (oc + 3< oc_edge && (j_res + 3) < strideC_1)? ((tem + 3)):-1;
                bufferstoreC(&C30,&C31,&C32,&C33,&offset0_tmp,&offset1_tmp,&offset2_tmp,&offset3_tmp,&globalStoreC);
           }

}

void convolution_2d_fp32(hipStream_t stream, 
                const argument& result, 
                const argument& x,
                const argument& w,
                const std::vector<std::size_t> padding,
                const std::vector<std::size_t> stride,
                const std::vector<std::size_t> dilation,
                const int group)
{
    
    // 参数
    unsigned int batch = x.get_shape().lens()[0];
    unsigned int channels = x.get_shape().lens()[1];
    unsigned int num_kernel = w.get_shape().lens()[0];
	unsigned int height = x.get_shape().lens()[2];
    unsigned int width = x.get_shape().lens()[3];
    unsigned int pad_h = padding[0];
	unsigned int pad_w = padding[1];
    unsigned int r = w.get_shape().lens()[2]; // 卷积核大小
    unsigned int s = w.get_shape().lens()[3];
	unsigned int stride_h =stride[0];
	unsigned int stride_w =stride[1];
    unsigned int dilate_h =dilation[0];
	unsigned int dilate_w =dilation[1];

    int dilate_filter_h = dilate_h * (r - 1) + 1;
	int dilate_filter_w = dilate_w * (s - 1) + 1;

	int o_w = (width  + 2*pad_w - dilate_filter_w)/stride_w  + 1;
	int o_h = (height + 2*pad_h - dilate_filter_h)/stride_h  + 1 ;

    int M = num_kernel/group;
    int N = o_h*o_w;
    int K = (channels/group)*r*s;
    unsigned int numgroupM = ((M + THREADS_NUM_M*T_M - 1)/(THREADS_NUM_M*T_M));
    unsigned int numgroupN = (N + THREADS_NUM_N*T_N -1)/(THREADS_NUM_N*T_N);
    unsigned int numgroupZ = group *batch ;

    int last_k_res_mask_bit = 0;
	int last_k_res =  (K%16);


	for(int i = 0;i < last_k_res;i++)
	{
       SetBit(last_k_res_mask_bit,i);
	}
    
      
    unsigned int strideA_0 =  (channels/group)*r*s;
    unsigned int strideA_1 = r;
    unsigned int strideA_2 = s;
    unsigned int strideA_3 = r*s;//为了复用这个寄存器
    unsigned int strideA_4 = num_kernel*(channels/group)*r*s;
    unsigned int strideB_0 = (channels/group)*width*height;
    unsigned int strideB_1 = width*height;
    unsigned int strideB_2 = width;
    unsigned int strideB_3 = height;//为了复用这个寄存器
    unsigned int strideB_4 = batch*channels*width*height;
    unsigned int strideC_0 = num_kernel*o_w*o_h;
    unsigned int strideC_1 = o_w*o_h;
    unsigned int strideC_2 = o_w;
    unsigned int strideC_3 = num_kernel;//为了复用这个寄存器
    unsigned int strideC_4 = o_h;//为了复用这个寄存器
    unsigned int remdinergroupX = M%(THREADS_NUM_M*T_M);
    unsigned int remdinergroupY = N%(THREADS_NUM_N*T_N);

    unsigned int group_out_channels = num_kernel/group;
    unsigned int group_block_nums = (group_out_channels - 1)/64 + 1;


    dim3 threads(THREADS_NUM_M,THREADS_NUM_N,1);
    dim3 groups(numgroupM, numgroupN, numgroupZ);

    hipLaunchKernelGGL(ImplicitGemmCommDilateGroupTA,
                                            groups,
                                            threads,
                                            0,
                                            stream,
                                            (float *)w.data(), 
                                            (float *)x.data(), 
                                            (float *)result.data(), 
                                            strideA_0,
                                            strideA_1,
                                            strideA_2,
                                            strideA_3,
                                            strideA_4,
                                            strideB_0,
                                            strideB_1,
                                            strideB_2,
                                            strideB_3,
                                            strideB_4,
                                            strideC_0,
                                            strideC_1,
                                            strideC_2,
                                            strideC_3,
                                            strideC_4,
                                            last_k_res_mask_bit,     
                                            last_k_res,     
                                            stride_h,  
                                            stride_w,
                                            pad_h,
                                            pad_w,
                                            dilate_h,
                                            dilate_w,
                                            group,
                                            group_out_channels,
                                            group_block_nums,
                                            numgroupM,
                                            numgroupN,
                                            remdinergroupX,
                                            remdinergroupY);
}

//////////////////////////// im2col ///////////////////////////////
shape compute_col_shape(const shape &input,const shape &weights,
                            const std::vector<std::size_t> padding,
                            const std::vector<std::size_t> stride,
                            const std::vector<std::size_t> dilation)
{
    // 先计算col图像的shape
    auto input_channels = input.lens()[1];
    auto kernel_height  = weights.lens()[2];
    auto kernel_width   = weights.lens()[3];

    auto padding_h = 2 * padding[0];
    auto padding_w = 2 * padding[1];
    auto output_height = (input.lens()[2] - (1 + dilation[0] * (kernel_height - 1)) + padding_h) / stride[0] + 1;
    auto output_width  = (input.lens()[3] - (1 + dilation[1] * (kernel_width - 1)) + padding_w) / stride[1] + 1;

    auto channels_col = kernel_height * kernel_width * input_channels;
    
    // col图像的行：channels_col，列：output_height * output_width
    return {input.type(), {channels_col,output_height * output_width}};
}

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col) 
{
    // grid stride looping
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;index < n;index += blockDim.x * gridDim.x)
    {
        // 每个线程的多维索引，这里计算的是在输出tensor中的索引
        const int w_col = index % width_col; // w坐标
        const int h_index = index / width_col;
        const int h_col = h_index % height_col; // h坐标，这里的索引是按照填充之后的图来计算的
        const int c_im = h_index / height_col; // c坐标

        // 每个线程对应的滑动窗口在原图中的索引
        const int h_offset = h_col * stride_h - pad_h; // 注意，由于有padding的存在，所以索引需要减去padding
        const int w_offset = w_col * stride_w - pad_w;

        // 该线程对应的输入tensor的数据指针
        const Dtype* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset; // 要拷贝的原图像起始位置

        // 该线程对应的输出tensor的数据指针
        const int c_col = c_im * kernel_h * kernel_w;// 因为是按照每个通道展开，所以这里需要乘以kernel_h * kernel_w
        Dtype* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        for (int i = 0; i < kernel_h; ++i) 
        {
            for (int j = 0; j < kernel_w; ++j) 
            {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                    data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
                data_col_ptr += height_col * width_col;// 步长为height_col * width_col，即每个滑动窗口展开为一列
            }
        }
    }
}

template <typename Dtype>
void im2col_gpu(hipStream_t stream,const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) 
{
  int height_col = (height + 2 * pad_h -(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  
  // 启动 channels * height_col * width_col 个线程
  int num_kernels = channels * height_col * width_col;
  int number_threads_per_block=256;
  dim3 threads(number_threads_per_block);
  dim3 blocks((num_kernels + number_threads_per_block - 1) / number_threads_per_block);
  im2col_gpu_kernel<Dtype><<<blocks,threads,0,stream>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
}


void convolution_2d_im2col(context& ctx, 
                const argument& result, 
                const argument& x,
                const argument& w,
                const argument& col_buf,
                const std::vector<std::size_t> padding,
                const std::vector<std::size_t> stride,
                const std::vector<std::size_t> dilation,
                const int group)
{
    // 对col图像进行reshape
    argument col_buffer=col_buf.reshape(compute_col_shape(x.get_shape(),w.get_shape(),padding,stride,dilation));

    int batch_size=x.get_shape().lens()[0];
    int number_of_one_batch_input=x.get_shape().lens()[1]*x.get_shape().lens()[2]*x.get_shape().lens()[3];
    int number_of_one_batch_output=result.get_shape().lens()[1]*result.get_shape().lens()[2]*result.get_shape().lens()[3];

    if(x.get_shape().type()==shape::float_type)
    {
        for(int i=0;i<batch_size;++i)
        {
            // 每次只处理batch=1的数据,防止内存溢出
            im2col_gpu(ctx.get_stream().get(),
                    (float *)x.data()+i*number_of_one_batch_input,
                    /*const int channels*/x.get_shape().lens()[1],
                    /*const int height*/x.get_shape().lens()[2],
                    /*const int width*/x.get_shape().lens()[3],
                    /*const int kernel_h*/w.get_shape().lens()[2], 
                    /*const int kernel_w*/w.get_shape().lens()[3],
                    /*const int pad_h*/padding[0], 
                    /*const int pad_w*/padding[1],
                    /*const int stride_h*/stride[0], 
                    /*const int stride_w*/stride[1],
                    /*const int dilation_h*/dilation[0], 
                    /*const int dilation_w*/dilation[1],
                    (float *)col_buffer.data());
            
            float alpha=1.0;
            float beta=0.0;

            // 计算一个batch的卷积
            argument A=w.reshape(shape{w.get_shape().type(),{w.get_shape().lens()[0],w.get_shape().lens()[1]*w.get_shape().lens()[2]*w.get_shape().lens()[3]}});
            argument B=col_buffer;
            argument C=argument{shape{A.get_shape().type(),{A.get_shape().lens()[0],B.get_shape().lens()[1]}},(float*)result.data()+i*number_of_one_batch_output};
            for(int g=0;g<group;++g)
            {
                argument A1=argument{shape{A.get_shape().type(),{A.get_shape().lens()[0]/group,A.get_shape().lens()[1]}},(float*)A.data()+g*A.get_shape().lens()[1]};
                argument B1=argument{shape{B.get_shape().type(),{B.get_shape().lens()[0]/group,B.get_shape().lens()[1]}},(float*)B.data()+g*B.get_shape().lens()[1]*B.get_shape().lens()[0]/group};
                argument C1=argument{shape{C.get_shape().type(),{C.get_shape().lens()[0]/group,C.get_shape().lens()[1]}},(float*)C.data()+g*C.get_shape().lens()[1]};
                gemm(ctx, C1.get_shape(), {A1,B1,C1}, alpha, beta, true, false);
            }
        }
    }
    else if(x.get_shape().type()==shape::half_type)
    {
        for(int i=0;i<batch_size;++i)
        {
            // 每次只处理batch=1的数据,防止内存溢出
            im2col_gpu(ctx.get_stream().get(),
                    (_Float16 *)x.data()+i*number_of_one_batch_input,
                    /*const int channels*/x.get_shape().lens()[1],
                    /*const int height*/x.get_shape().lens()[2],
                    /*const int width*/x.get_shape().lens()[3],
                    /*const int kernel_h*/w.get_shape().lens()[2], 
                    /*const int kernel_w*/w.get_shape().lens()[3],
                    /*const int pad_h*/padding[0], 
                    /*const int pad_w*/padding[1],
                    /*const int stride_h*/stride[0], 
                    /*const int stride_w*/stride[1],
                    /*const int dilation_h*/dilation[0], 
                    /*const int dilation_w*/dilation[1],
                    (_Float16 *)col_buffer.data());
            
            float alpha=1.0;
            float beta=0.0;

            // 计算一个batch的卷积
            argument A=w.reshape(shape{w.get_shape().type(),{w.get_shape().lens()[0],w.get_shape().lens()[1]*w.get_shape().lens()[2]*w.get_shape().lens()[3]}});
            argument B=col_buffer;
            argument C=argument{shape{A.get_shape().type(),{A.get_shape().lens()[0],B.get_shape().lens()[1]}},(_Float16*)result.data()+i*number_of_one_batch_output};
            for(int g=0;g<group;++g)
            {
                argument A1=argument{shape{A.get_shape().type(),{A.get_shape().lens()[0]/group,A.get_shape().lens()[1]}},(_Float16*)A.data()+g*A.get_shape().lens()[1]};
                argument B1=argument{shape{B.get_shape().type(),{B.get_shape().lens()[0]/group,B.get_shape().lens()[1]}},(_Float16*)B.data()+g*B.get_shape().lens()[1]*B.get_shape().lens()[0]/group};
                argument C1=argument{shape{C.get_shape().type(),{C.get_shape().lens()[0]/group,C.get_shape().lens()[1]}},(_Float16*)C.data()+g*C.get_shape().lens()[1]};
                gemm(ctx, C1.get_shape(), {A1,B1,C1}, alpha, beta, true, false);
            }

        }

    }
    


}


} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
