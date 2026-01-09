# Intended to be run from nvidia/cuda:13.1.0-devel-ubuntu24.04
.PHONY: all clean

all: tc-ptx tc

cpu_mul: cpu_mul.c
	gcc -O2 -std=c23 -mavx512bf16 -o cpu_mul cpu_mul.c

tc-ptx: tc.ptx
	ptxas --gpu-name sm_90a --output-file tc.cubin tc.ptx

tc: tc.cu tc-ptx
	nvcc -O2 --compiler-options=-mavx512bf16 -lcuda -std=c++20 -o tc tc.cu

clean:
	rm -f cpu_mul tc.cubin