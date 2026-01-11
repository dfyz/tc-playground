#pragma once

#include "common.h"

#include <array>

using GemmOutput = std::array<std::array<float, kN>, kM>;

GemmOutput MulMatMatHopper(const MatA& mat_a, const MatB& mat_b);
