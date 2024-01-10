# install
rbuild build -d depend --cxx=${ROCM_PATH}/llvm/bin/clang++ --cc=${ROCM_PATH}/llvm/bin/clang

# test
# cd build
# make check -j48

# 打包生成rpm文件
# cd build
# yum install rpm-build -y
# cmake ../
# make package -j48
