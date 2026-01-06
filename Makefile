.PHONY: all clean

all: cpu_mul tc-ptx tc

cpu_mul: cpu_mul.c
	gcc -O2 -std=c23 -mavx512bf16 -o cpu_mul cpu_mul.c

tc-ptx: tc.ptx
	ptxas --gpu-name sm_90a --output-file tc.cubin tc.ptx

tc: tc.cu tc-ptx
	nvcc -O2 -std=c++20 -lcuda -o tc tc.cu

clean:
	rm -f cpu_mul tc.cubin