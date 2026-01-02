cpu_mul: cpu_mul.c
	gcc -O2 -std=c23 -mavx512bf16 -o cpu_mul cpu_mul.c

tc-ptx: tc.ptx
	ptxas --gpu-name sm_90a --output-file tc.cubin tc.ptx