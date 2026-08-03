#include "gemm_hopper_emu_2.h"

#include <math.h>

constexpr auto FP32_EXP_BIAS = 127;
constexpr auto FP32_MANTISSA_BITS = 23;
constexpr auto FP64_MANTISSA_BITS = 52;
constexpr auto FRAC_SUM_BITS = 25;

union fp32_int {
    float    f;
    uint32_t i;

    struct parts {
        unsigned frac    : 23;
        unsigned exponent: 8;
        unsigned sign    : 1;
    } p;
};

union fp64_int {
    double   f;
    uint64_t i;
};

struct addend {
    float frac;
    int   exponent;
};

float load_bf16(const uint16_t* ptr) {
    union fp32_int res = {.i = *ptr};
    res.i <<= 16;
    return res.f;
}

int32_t unbias_exp(uint32_t exp) {
    return (int32_t)exp - FP32_EXP_BIAS;
}

uint32_t bias_exp(int32_t exp) {
    return (uint32_t)(exp + FP32_EXP_BIAS);
}

void decompose(float x, float* frac, int32_t* exponent) {
    union fp32_int parts = {.f = x};
    int32_t m_exp = 0;
    int32_t real_exp = unbias_exp(parts.p.exponent);
    if (real_exp == -FP32_EXP_BIAS) {
        ++real_exp;
        parts.f = ldexpf(parts.f, FP32_MANTISSA_BITS);
        m_exp = (unbias_exp(parts.p.exponent) - FP32_MANTISSA_BITS) - real_exp;
    }
    parts.p.exponent = bias_exp(m_exp);
    *frac = parts.f;
    *exponent = real_exp;
}

double chop_frac(double x, int n_bits) {
    union fp64_int res = {.f = x};
    int shift_by = FP64_MANTISSA_BITS - n_bits;
    res.i = res.i >> shift_by << shift_by;
    return res.f;
}

double truncate_addend(double frac, int32_t cur_exp, int32_t max_exp) {
    return chop_frac(
        ldexp(frac, cur_exp - max_exp),
        FRAC_SUM_BITS
    );
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
