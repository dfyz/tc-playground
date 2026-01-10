#pragma once

#include <array>
#include <cuda_bf16.h>

constexpr size_t kK = 16;
constexpr size_t kM = 64;
constexpr size_t kN = 256;

using Vec = std::array<__nv_bfloat16, kK>;
using MatA  = std::array<Vec, kM>;
using MatB  = std::array<Vec, kN>;
