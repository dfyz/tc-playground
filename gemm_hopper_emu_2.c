#include "gemm_hopper_emu_2.h"

#include <math.h>

constexpr auto FP32_EXP_BIAS = 127;
constexpr auto FP64_EXP_BIAS = 1023;

constexpr auto FP32_MANTISSA_BITS = 23;
constexpr auto FP64_MANTISSA_BITS = 52;

constexpr auto FRAC_SUM_BITS = 25;

union fp32_int {
    float    f;
    uint32_t i;

    struct {
        uint32_t frac    : FP32_MANTISSA_BITS;
        uint32_t exponent: 8;
        uint32_t sign    : 1;
    } p;
};

int32_t extract_fp32_exp(union fp32_int x) {
    return (int32_t)x.p.exponent - FP32_EXP_BIAS;
}

uint32_t store_fp32_exp(union fp32_int* x, int32_t e) {
    x->p.exponent = e + FP32_EXP_BIAS;
}

union fp64_int {
    double   f;
    uint64_t i;

    struct {
        uint64_t frac    : FP64_MANTISSA_BITS;
        uint32_t exponent: 11;
        uint32_t sign    : 1;
    } p;
};

int32_t extract_fp64_exp(union fp64_int x) {
    return (int32_t)x.p.exponent - FP64_EXP_BIAS;
}

struct addend {
    float frac;
    int   exponent;
};

float load_bf16(const uint16_t* ptr) {
    union fp32_int res = {.i = *ptr};
    res.i <<= 16;
    return res.f;
}

void decompose(float x, float* frac, int32_t* exponent) {
    union fp32_int parts = {.f = x};
    int32_t real_exp = extract_fp32_exp(parts);
    int32_t m_exp = 0;
    if (real_exp == -FP32_EXP_BIAS) {
        ++real_exp;
        parts.f *= (float)(1u << FP32_MANTISSA_BITS);
        m_exp = (extract_fp32_exp(parts) - FP32_MANTISSA_BITS) - real_exp;
    }
    store_fp32_exp(&parts, m_exp);
    *frac = parts.f;
    *exponent = real_exp;
}

double chop_frac(double x, int n_bits) {
    union fp64_int res = {.f = x};
    if (n_bits < 0) {
        n_bits = 0;
    }
    if (n_bits > FP64_MANTISSA_BITS) {
        n_bits = FP64_MANTISSA_BITS;
    }
    int shift_by = FP64_MANTISSA_BITS - n_bits;
    res.i = res.i >> shift_by << shift_by;
    return res.f;
}

double truncate_addend(double frac, int32_t cur_exp, int32_t max_exp) {
    int bits_delta = extract_fp64_exp((union fp64_int){.f = frac});
    int32_t exp_delta = max_exp - cur_exp;
    double aligned = ldexp(frac, -exp_delta);
    int n_chop = FRAC_SUM_BITS - exp_delta + bits_delta;
    return chop_frac(aligned, n_chop);
}

float MulVecVecHopperEmu2(float c, const uint16_t vec_a[VEC_K], const uint16_t vec_b[VEC_K]) {
    struct addend addends[VEC_K];
    int32_t max_exp = -133;

    for (size_t ii = 0; ii < VEC_K; ++ii) {
        float   a_frac, b_frac;
        int32_t a_exp, b_exp;
        decompose(load_bf16(vec_a + ii), &a_frac, &a_exp);
        decompose(load_bf16(vec_b + ii), &b_frac, &b_exp);

        addends[ii].frac = a_frac * b_frac;
        addends[ii].exponent = a_exp + b_exp;
        if (addends[ii].exponent > max_exp) {
            max_exp = addends[ii].exponent;
        }
    }

    float   c_frac;
    int32_t c_exp;
    decompose(c, &c_frac, &c_exp);
    if (c != 0.0f && c_exp > max_exp) {
        max_exp = c_exp;
    }

    double frac_sum = truncate_addend(c_frac, c_exp, max_exp);
    for (size_t ii = 0; ii < VEC_K; ++ii) {
        frac_sum += truncate_addend(addends[ii].frac, addends[ii].exponent, max_exp);
    }
    return chop_frac(
        ldexp(frac_sum, max_exp),
        FP32_MANTISSA_BITS
    );
}
