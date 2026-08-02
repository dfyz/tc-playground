#pragma once

#include <stddef.h>
#include <stdint.h>

constexpr size_t VEC_K = 16;

#ifdef __cplusplus
extern "C"
#endif
float MulVecVecHopperEmu2(float c, const uint32_t vec_a[VEC_K], const uint32_t vec_b[VEC_K]);
