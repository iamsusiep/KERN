#!/usr/bin/env bash
cuda_path=/usr/local/cuda
cd src/cuda
echo "Compiling stnn kernels by nvcc..."
/usr/local/cuda/bin/nvcc -c -o nms.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../
python build.py
