#include "gemm_hopper_emu_2.h"

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#include <fenv.h>

constexpr int FP64_MANTISSA_BITS = 52;
constexpr int FP64_EXP_BIAS      = 1023;
constexpr int ZERO_EXP           = -133;
constexpr int MAGIC_SHIFT        = 27;

union fp32_int {
    float    f;
    uint32_t i;
};

union fp64_int {
    double   f;
    uint64_t i;
    struct {
        uint64_t frac:     FP64_MANTISSA_BITS;
        uint32_t exponent: 11;
        uint32_t sign:     1;
    }        p;
};

double load_bf16(const uint16_t* ptr) {
    union fp32_int res = {.i = *ptr};
    res.i <<= 16;
    return res.f;
}

int32_t get_exp(double x) {
    if (x == 0.0) {
        return ZERO_EXP;
    }
    union fp64_int tmp = {.f = x};
    return (int32_t)tmp.p.exponent - FP64_EXP_BIAS;
}

double shift(double x, double magic, int32_t max_exp) {
    magic = x < 0 ? -magic : magic;
    return x + magic - magic;
}

int32_t max(int32_t a, int32_t b) {
    return a > b ? a : b;
}

float tc_bf16_fp32(float c, const uint16_t vec_a[VEC_K], const uint16_t vec_b[VEC_K]) {
    double addends[VEC_K];
    int32_t max_exp = get_exp(c);

    for (size_t ii = 0; ii < VEC_K; ++ii) {
        double lhs = load_bf16(vec_a + ii);
        double rhs = load_bf16(vec_b + ii);
        addends[ii] = lhs * rhs;
        max_exp = max(max_exp, get_exp(lhs) + get_exp(rhs));
    }

    union fp64_int magic = {.p = {
        .frac     = 0,
        .exponent = FP64_EXP_BIAS + max_exp + MAGIC_SHIFT,
        .sign     = 0,
    }};

    double res = shift(c, magic.f, max_exp);
    for (size_t ii = 0; ii < VEC_K; ++ii) {
        res += shift(addends[ii], magic.f, max_exp);
    }
    return res;
}

float MulVecVecHopperEmu2(float c, const uint16_t vec_a[VEC_K], const uint16_t vec_b[VEC_K]) {
    int old_mode = fegetround();
    fesetround(FE_TOWARDZERO);
    float res = tc_bf16_fp32(c, vec_a, vec_b);
    fesetround(old_mode);
    return res;
}
