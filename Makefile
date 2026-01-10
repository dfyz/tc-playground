# Intended to be run from nvidia/cuda:13.1.0-devel-ubuntu24.04
CXX       ?= g++
PTXAS     ?= ptxas
CUDA_HOME ?= /usr/local/cuda

CFLAGS  += -I$(CUDA_HOME)/include -std=c++20 -O2
LDFLAGS += -L$(CUDA_HOME)/lib64

.PHONY: all clean

all: tc

tc-ptx: tc.ptx
	$(PTXAS) --gpu-name sm_90a --output-file tc.cubin tc.ptx

gemm-avx512:
	$(CXX) $(CFLAGS) -mavx512bf16 -c gemm_avx512.cpp

gemm-hopper:
	$(CXX) $(CFLAGS) -c gemm_hopper.cpp

tc: tc.cpp tc-ptx gemm-avx512 gemm-hopper
	$(CXX) $(CFLAGS) $(LDFLAGS) -O2 -std=c++20 \
		-o tc \
		tc.cpp gemm_avx512.o gemm_hopper.o \
		-lcuda

clean:
	rm -f tc *.cubin *.o
