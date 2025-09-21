#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

float bf16_to_fp32(uint16_t x) {
    unsigned char res[sizeof(float)] = { 0 };
    memcpy(res + sizeof(x), &x, sizeof(x));
    return *(float*)res;
}

uint32_t fp32_to_u32(float x) {
    unsigned char res[sizeof(uint32_t)];
    memcpy(res, &x, sizeof(x));
    return *(uint32_t*)res;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: BF16_HEX BF16_HEX\n");
        exit(1);
    }

    uint16_t a = strtoul(argv[1], NULL, 16);
    uint16_t b = strtoul(argv[2], NULL, 16);

    printf("a = 0x%04hX     %g\n", a, bf16_to_fp32(a));
    printf("b = 0x%04hX     %g\n", b, bf16_to_fp32(b));

    __m512bh a_vec = (__m512bh)_mm512_mask_set1_epi16(_mm512_setzero_epi32(), 1, a);
    __m512bh b_vec = (__m512bh)_mm512_mask_set1_epi16(_mm512_setzero_epi32(), 1, b);

    __m512 d_vec = _mm512_dpbf16_ps(_mm512_setzero(), a_vec, b_vec);
    float d = _mm512_cvtss_f32(d_vec);
    printf("d = 0x%08X %g\n", fp32_to_u32(d), d);
}