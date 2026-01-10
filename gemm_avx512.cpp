#include "gemm_avx512.h"

#include <immintrin.h>

float MulVecVecAvx512(const Vec& vec_a, const Vec& vec_b) {
    __m512bh a_pack = (__m512bh)_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)vec_a.data()));
    __m512bh b_pack = (__m512bh)_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)vec_b.data()));
    __m512 packed_prods = _mm512_dpbf16_ps(_mm512_setzero(), a_pack, b_pack);
    alignas(64) float prods[16];
    _mm512_store_ps(prods, packed_prods);
    float res = 0.0f;
    for (auto x : prods) {
        res += x;
    }
    return res;
}