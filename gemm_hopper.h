#pragma once

#include "common.h"

#include <memory>

struct GemmOutput {
    virtual float operator()(size_t row, size_t col) const = 0;
};

std::unique_ptr<GemmOutput> MulMatMatHopper(const MatA& mat_a, const MatB& mat_b);
