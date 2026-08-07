#include "gemm_hopper_emu_2.h"

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#include <fenv.h>

constexpr int FP64_MANTISSA_BITS = 52;
constexpr int FP64_EXP_BIAS      = 1023;

constexpr int FP32_MANTISSA_BITS = 23;
constexpr int FP32_EXP_BIAS      = 127;

union fp32_int {
    float    f;
    uint32_t i;
};

union fp64_int {
    double   f;
    uint64_t i;
};

float load_bf16(const uint16_t* ptr) {
    return (union fp32_int){
        .i = *ptr << 16,
    }.f;
}

int32_t get_exp(float x) {
    union fp32_int tmp = {.f = x};
    int32_t biased_exp = (tmp.i >> FP32_MANTISSA_BITS) & 0xff;
    return (biased_exp + (biased_exp == 0)) - FP32_EXP_BIAS;
}

double shift(double x, double magic) {
    magic = x < 0 ? -magic : magic;
    return x + magic - magic;
}

int32_t guard_zero(double res, int32_t exp) {
    return res == 0.0 ? -133 : exp;
}

float tc_bf16_fp32(float c, const uint16_t vec_a[VEC_K], const uint16_t vec_b[VEC_K]) {
    double addends[VEC_K];
    int32_t max_exp = guard_zero(c, get_exp(c));

    for (size_t ii = 0; ii < VEC_K; ++ii) {
        float lhs   = load_bf16(vec_a + ii);
        float rhs   = load_bf16(vec_b + ii);
        double prod = (double)lhs * (double)rhs;
        int32_t cur_exp = guard_zero(prod, get_exp(lhs) + get_exp(rhs));
        if (cur_exp > max_exp) {
            max_exp = cur_exp;
        }
        addends[ii] = prod;
    }

    union fp64_int magic = {.f = 0x1p27};
    magic.i += (uint64_t)max_exp << FP64_MANTISSA_BITS;

    double res = shift(c, magic.f);
    for (size_t ii = 0; ii < VEC_K; ++ii) {
        res += shift(addends[ii], magic.f);
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
