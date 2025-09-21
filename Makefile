cpu_mul: cpu_mul.c
	gcc -O2 -std=c23 -mavx512bf16 -o cpu_mul cpu_mul.c