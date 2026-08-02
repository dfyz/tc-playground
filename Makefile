# Intended to be run from nvidia/cuda:13.1.0-devel-ubuntu24.04
CC        ?= gcc
CXX       ?= g++
PTXAS     ?= ptxas
CUDA_HOME ?= /usr/local/cuda

CC_STD  = -std=c2x
CXX_STD = -std=c++20

override CFLAGS  += -I$(CUDA_HOME)/include -O2
override LDFLAGS += -L$(CUDA_HOME)/lib64

.PHONY: all clean

all: tc reproduce_tc

tc-ptx: tc.ptx
	$(PTXAS) --gpu-name sm_90a --output-file tc.cubin tc.ptx

gemm-avx512:
	$(CXX) $(CFLAGS) $(CXX_STD) -mavx512bf16 -c gemm_avx512.cpp

gemm-hopper:
	$(CXX) $(CFLAGS) $(CXX_STD) -c gemm_hopper.cpp

gemm-hopper-emu:
	$(CXX) $(CFLAGS) $(CXX_STD) -c gemm_hopper_emu.cpp

gemm-hopper-emu-2:
	$(CC) $(CFLAGS) $(CC_STD) -c gemm_hopper_emu_2.c

tc: tc.cpp tc-ptx gemm-avx512 gemm-hopper gemm-hopper-emu gemm-hopper-emu-2
	$(CXX) $(CFLAGS) $(CXX_STD) $(LDFLAGS) -O2 \
		-o tc \
		tc.cpp gemm_avx512.o gemm_hopper.o gemm_hopper_emu.o gemm_hopper_emu_2.o \
		-lcuda

reproduce_tc: reproduce_tc.cpp gemm_hopper_emu.o
	$(CXX) $(CFLAGS) $(CXX_STD) $(LDFLAGS) -O2 \
		-o reproduce_tc \
		reproduce_tc.cpp gemm_hopper_emu.o \
		-lcuda

clean:
	rm -f tc *.cubin *.o
