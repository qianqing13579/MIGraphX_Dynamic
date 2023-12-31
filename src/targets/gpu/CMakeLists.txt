#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} $ENV{ROCM_PATH}/hip $ENV{ROCM_PATH}/hcc)
find_package(miopen)

# rocblas
find_package(rocblas REQUIRED PATHS $ENV{ROCM_PATH})
message(STATUS "Build with rocblas")

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

include(Embed)
file(GLOB KERNEL_FILES ${CONFIGURE_DEPENDS}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/migraphx/kernels/*.hpp)
message(STATUS "KERNEL_FILES: ${KERNEL_FILES}")
add_embed_library(migraphx_kernels ${KERNEL_FILES})

add_library(migraphx_device
    device/acos.cpp
    device/acosh.cpp
    device/add.cpp
    device/add_clip.cpp
    device/add_relu.cpp
    device/add_sigmoid.cpp
    device/add_tanh.cpp
    device/argmax.cpp
    device/argmin.cpp
    device/asin.cpp
    device/asinh.cpp
    device/atan.cpp
    device/atanh.cpp
    device/ceil.cpp
    device/clip.cpp
    device/concat.cpp
    device/contiguous.cpp
    device/convert.cpp
    device/convolution_2d_fp32.cpp
    device/convolution_2d_fp16.cpp
    device/cos.cpp
    device/cosh.cpp
    device/div.cpp
    device/equal.cpp
    device/erf.cpp
    device/exp.cpp
    device/fill.cpp
    device/floor.cpp
    device/gather.cpp
    device/gelu.cpp
    device/greater.cpp
    device/int8_gemm_pack.cpp
    device/layernorm.cpp
    device/less.cpp
    device/log.cpp
    device/logical_and.cpp
    device/logical_or.cpp
    device/logical_xor.cpp
    device/logsoftmax.cpp
    device/max.cpp
    device/min.cpp
    device/mul.cpp
    device/mul_add.cpp
    device/mul_add_relu.cpp
    device/multinomial.cpp
    device/mod.cpp
    device/nonzero.cpp
    device/pad.cpp
    device/pad_dynamic.cpp
    device/pow.cpp
    device/prelu.cpp
    device/prefix_scan_sum.cpp
    device/recip.cpp
    device/reduce_max.cpp
    device/reduce_mean.cpp
    device/reduce_min.cpp
    device/reduce_sum.cpp
    device/reduce_prod.cpp
    device/relu.cpp
    device/reverse.cpp
    device/rnn_variable_seq_lens.cpp
    device/round.cpp
    device/rsqrt.cpp
    device/scatter.cpp
    device/scatternd_none.cpp
    device/scatter_elements.cpp
    device/selu.cpp
    device/Shape.cpp
    device/sigmoid.cpp
    device/sign.cpp
    device/sin.cpp
    device/sinh.cpp
    device/softmax.cpp
    device/sqdiff.cpp
    device/sqrt.cpp
    device/sub.cpp
    device/tan.cpp
    device/tanh.cpp
    device/tile.cpp
    device/topk.cpp
    device/unary_not.cpp
    device/upsample.cpp
    device/where.cpp
)
add_library(compile_for_gpu INTERFACE)
target_compile_options(compile_for_gpu INTERFACE -std=c++17 -fno-gpu-rdc -Wno-cuda-compat -Wno-unused-command-line-argument -Xclang -fallow-half-arguments-and-returns)
target_link_libraries(compile_for_gpu INTERFACE hip::device -fno-gpu-rdc -Wno-invalid-command-line-argument -Wno-unused-command-line-argument -Wno-option-ignored)
check_cxx_compiler_flag("--cuda-host-only -fhip-lambda-host-device -x hip" HAS_HIP_LAMBDA_HOST_DEVICE)
if(HAS_HIP_LAMBDA_HOST_DEVICE)
  message(STATUS "Enable -fhip-lambda-host-device")
  target_compile_options(compile_for_gpu INTERFACE -fhip-lambda-host-device)
endif()

set_target_properties(migraphx_device PROPERTIES EXPORT_NAME device)
rocm_set_soversion(migraphx_device ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_device)
target_link_libraries(migraphx_device PUBLIC migraphx)
target_link_libraries(migraphx_device PRIVATE compile_for_gpu)
target_include_directories(migraphx_device PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(migraphx_device PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/device/include>)

add_library(kernel_file_check EXCLUDE_FROM_ALL)
foreach(KERNEL_FILE ${KERNEL_FILES})
    get_filename_component(KERNEL_BASE_FILE ${KERNEL_FILE} NAME_WE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/kernels/include/migraphx/kernels/${KERNEL_BASE_FILE}.cpp "#include <migraphx/kernels/${KERNEL_BASE_FILE}.hpp>\n")
    target_sources(kernel_file_check PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/kernels/include/migraphx/kernels/${KERNEL_BASE_FILE}.cpp)
endforeach()
target_compile_definitions(kernel_file_check PRIVATE -DMIGRAPHX_NLOCAL=256)
target_include_directories(kernel_file_check PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/>)
target_link_libraries(kernel_file_check compile_for_gpu)

rocm_clang_tidy_check(kernel_file_check)

file(GLOB JIT_GPU_SRCS ${CONFIGURE_DEPENDS} ${CMAKE_CURRENT_SOURCE_DIR}/jit/*.cpp)
add_library(migraphx_gpu
    abs.cpp
    analyze_streams.cpp
    allocation_model.cpp
    argmax.cpp
    argmin.cpp
    batch_norm_inference.cpp
    clip.cpp
    code_object_op.cpp
    compile_ops.cpp
    compile_gen.cpp
    compile_hip.cpp
    compile_hip_code_object.cpp
    compiler.cpp
    concat.cpp
    convert.cpp
    convolution.cpp
    convolution_dynamic.cpp
    constantofshape.cpp
    deconvolution.cpp
    device_name.cpp
    elu.cpp
    fuse_mlir.cpp
    fuse_ops.cpp
    gather.cpp
    gemm_impl.cpp
    hip.cpp
    int8_conv_pack.cpp
    int8_gemm_pack.cpp
    kernel.cpp
    lowering.cpp
    logsoftmax.cpp
    loop.cpp
    lrn.cpp
    leaky_relu.cpp
    lstm.cpp
    lstm_nomiopen.cpp
    mlir.cpp
    multinomial.cpp
    multibroadcast_dynamic.cpp
    nonzero.cpp
    pack_args.cpp
    pack_int8_args.cpp
    prefuse_ops.cpp
    pad.cpp
    pad_dynamic.cpp
    perfdb.cpp
    pooling.cpp
    quant_convolution.cpp
    range.cpp
    reverse.cpp
    reshape_dynamic.cpp
    resize.cpp
    rnn_variable_seq_lens.cpp
    rocblas.cpp
    scatter.cpp
    scatter_elements.cpp
    scatternd_none.cpp
    selu.cpp
    Shape.cpp
    slice_dynamic.cpp
    schedule_model.cpp
    softmax.cpp
    sync_device.cpp
    target.cpp
    tile.cpp
    topk.cpp
    upsample.cpp
    write_literals.cpp
    ${JIT_GPU_SRCS}
)
set_target_properties(migraphx_gpu PROPERTIES EXPORT_NAME gpu)

function(register_migraphx_gpu_ops PREFIX)
    foreach(OP ${ARGN})
        register_op(migraphx_gpu HEADER migraphx/gpu/${OP}.hpp OPERATORS gpu::${PREFIX}${OP} INCLUDES migraphx/gpu/context.hpp)
    endforeach()
endfunction()
register_migraphx_gpu_ops(hip_
    acosh
    acos
    add
    argmax
    argmin
    asinh
    asin
    atanh
    atan
    ceil
    clip
    concat
    convert
    constantofshape
    cosh
    cos
    div
    equal
    erf
    exp
    floor
    gather
    greater
    less
    log
    logsoftmax
    logical_and
    logical_or
    logical_xor
    loop
    max
    min
    mul
    multinomial
    multibroadcast_dynamic
    mod
    nonzero
    pad
    pad_dynamic
    pow
    prelu
    prefix_scan_sum
    range
    recip
    reduce_max
    reduce_mean
    reduce_min
    reduce_prod
    reduce_sum
    relu
    reverse
    reshape_dynamic
    resize
    round
    rsqrt
    scatter
    scatter_elements
    scatternd_none
    selu
    Shape
    sigmoid
    sign
    sinh
    sin
    slice_dynamic
    softmax
    sqdiff
    sqrt
    sub
    tanh
    tan
    tile
    topk
    unary_not
    upsample
    where
)
register_migraphx_gpu_ops(miopen_
    abs
    batch_norm_inference
    contiguous
    convolution
    convolution_dynamic
    deconvolution
    elu
    int8_conv_pack
    leaky_relu
    lrn
    lstm
    pooling
    quant_convolution
)
register_op(migraphx_gpu 
    HEADER migraphx/gpu/rnn_variable_seq_lens.hpp 
    OPERATORS gpu::hip_rnn_var_sl_shift_sequence gpu::hip_rnn_var_sl_shift_output gpu::hip_rnn_var_sl_last_output
    INCLUDES migraphx/gpu/context.hpp)
register_op(migraphx_gpu 
    HEADER migraphx/gpu/int8_gemm_pack.hpp 
    OPERATORS gpu::hip_int8_gemm_pack_a gpu::hip_int8_gemm_pack_b
    INCLUDES migraphx/gpu/context.hpp)
register_op(migraphx_gpu 
    HEADER migraphx/gpu/gemm.hpp 
    OPERATORS gpu::rocblas_gemm<op::dot> gpu::rocblas_gemm<op::quant_dot>
    INCLUDES migraphx/gpu/context.hpp)
rocm_set_soversion(migraphx_gpu ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_gpu)

# look for offload bundler
get_filename_component(CMAKE_CXX_COMPILER_PATH "${CMAKE_CXX_COMPILER}" PATH)
if(CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+$")
    find_program(MIGRAPHX_OFFLOADBUNDLER_BIN clang-offload-bundler
        HINTS ${CMAKE_CXX_COMPILER_PATH}
        PATH_SUFFIXES bin
        PATHS $ENV{ROCM_PATH}/llvm
    )
else()
    find_program(MIGRAPHX_EXTRACT_KERNEL extractkernel
        PATH_SUFFIXES bin
        HINTS ${CMAKE_CXX_COMPILER_PATH}
        PATHS
            $ENV{ROCM_PATH}/hip
            $ENV{ROCM_PATH}/hcc
            $ENV{ROCM_PATH}
    )
endif()

message(STATUS "clang-offload-bundler: ${MIGRAPHX_OFFLOADBUNDLER_BIN}")
message(STATUS "extractkernel: ${MIGRAPHX_EXTRACT_KERNEL}")

set(MIGRAPHX_ENABLE_MLIR OFF CACHE BOOL "")
if(MIGRAPHX_ENABLE_MLIR)
    # Find package rocMLIR
    find_package(rocMLIR 1.0.0 CONFIG REQUIRED)
    message(STATUS "Build with rocMLIR::rockCompiler ${rocMLIR_VERSION}")
    target_compile_definitions(migraphx_gpu PRIVATE "-DMIGRAPHX_MLIR")
    target_link_libraries(migraphx_gpu PUBLIC rocMLIR::rockCompiler)
endif()

set(MIGRAPHX_USE_HIPRTC OFF CACHE BOOL "")
if(MIGRAPHX_USE_HIPRTC)
target_compile_definitions(migraphx_gpu PRIVATE -DMIGRAPHX_USE_HIPRTC=1)
else()
# Get flags needed to compile hip
include(TargetFlags)
target_flags(HIP_COMPILER_FLAGS hip::device)
# Remove cuda arch flags
string(REGEX REPLACE --cuda-gpu-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
string(REGEX REPLACE --offload-arch=[a-z0-9:+-]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
# Skip library paths since hip will incorrectly treat it as a source file
string(APPEND HIP_COMPILER_FLAGS " ")
foreach(_unused RANGE 2)
    string(REGEX REPLACE " /[^ ]+\\.(a|so) " " " HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
endforeach()

message(STATUS "Hip compiler flags: ${HIP_COMPILER_FLAGS}")
target_compile_definitions(migraphx_gpu PRIVATE 
    "-DMIGRAPHX_HIP_COMPILER=${CMAKE_CXX_COMPILER}" 
    "-DMIGRAPHX_HIP_COMPILER_FLAGS=${HIP_COMPILER_FLAGS}" 
    "-DMIGRAPHX_OFFLOADBUNDLER_BIN=${MIGRAPHX_OFFLOADBUNDLER_BIN}"
    "-DMIGRAPHX_EXTRACT_KERNEL=${MIGRAPHX_EXTRACT_KERNEL}"
    "-DMIGRAPHX_USE_HIPRTC=0"
)
if(DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
execute_process(COMMAND which ${CMAKE_CXX_COMPILER_LAUNCHER} OUTPUT_VARIABLE MIGRAPHX_HIP_COMPILER_LAUNCHER)
string(STRIP "${MIGRAPHX_HIP_COMPILER_LAUNCHER}" MIGRAPHX_HIP_COMPILER_LAUNCHER)
target_compile_definitions(migraphx_gpu PRIVATE "-DMIGRAPHX_HIP_COMPILER_LAUNCHER=${MIGRAPHX_HIP_COMPILER_LAUNCHER}")
endif()

endif()

# Check miopen find mode api
include(CheckLibraryExists)
get_target_property(MIOPEN_LOCATION MIOpen LOCATION)
check_library_exists(MIOpen "miopenHiddenSetConvolutionFindMode" "${MIOPEN_LOCATION}" HAS_FIND_MODE_API)
if(HAS_FIND_MODE_API)
    target_compile_definitions(migraphx_gpu PUBLIC -DMIGRAPHX_HAS_FIND_MODE_API)
    message(STATUS "MIOpen has find mode api")
else()
    message(STATUS "MIOpen does not have find mode api")
endif()

# Workaround broken rocblas headers
target_compile_definitions(migraphx_gpu PUBLIC -D__HIP_PLATFORM_HCC__=1)
target_link_libraries(migraphx_gpu PUBLIC migraphx MIOpen roc::rocblas)
target_link_libraries(migraphx_gpu PRIVATE migraphx_device migraphx_kernels)

add_subdirectory(driver)

rocm_install_targets(
  TARGETS migraphx_gpu migraphx_device compile_for_gpu
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

