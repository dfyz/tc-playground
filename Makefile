# Intended to be run from nvidia/cuda:13.1.0-devel-ubuntu24.04
CUDA_HOME ?= /usr/local/cuda

CFLAGS  += -I$(CUDA_HOME)/include
LDFLAGS += -L$(CUDA_HOME)/lib64
LDLIBS  += -lcuda

.PHONY: all clean

all: tc

tc-ptx: tc.ptx
	ptxas --gpu-name sm_90a --output-file tc.cubin tc.ptx

mvv-avx512:
	g++ $(CFLAGS) -O2 -std=c++20 -mavx512bf16 -c -o mvv_avx512.o mvv_avx512.cpp

tc: tc.cpp tc-ptx mvv-avx512
	g++ $(CFLAGS) $(LDFLAGS) -O2 -std=c++20 -o tc tc.cpp mvv_avx512.o -lcuda

clean:
	rm -f tc *.cubin *.o