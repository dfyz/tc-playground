#include "common.h"
#include "gemm_hopper_emu.h"

#include <cstdio>
#include <err.h>
#include <vector>

#include <cuda_bf16.h>

constexpr size_t kFullM = 8192;
constexpr size_t kFullN = 12288;
constexpr size_t kFullK = 2048;

std::vector<Vec> ParseFile(const char* file_name, size_t rows, size_t cols) {
    std::vector<Vec> result(rows * (cols / kK));
    FILE* f = fopen(file_name, "r");
    if (f == nullptr) {
        errx(1, "failed to open %s", file_name);
    }

    for (size_t ii = 0; ii < result.size(); ++ii) {
        if (fread(result[ii].data(), sizeof(__nv_bfloat16), kK, f) != kK) {
            errx(1, "failed to read vector %zu", ii);
        }
    }

    fclose(f);
    return result;
}

size_t GetPos(size_t row, size_t col, size_t cols) {
    return row * cols + col;
}

int main() {
    const auto a_mat = ParseFile("a_prefill.bin", kFullM, kFullK);
    const auto b_mat = ParseFile("b_prefill.bin", kFullN, kFullK);
    const auto c_mat = ParseFile("c_prefill.bin", kFullM, kFullN);

    for (size_t row = 0; row < kFullM; ++row) {
        printf("%zu/%zu\n", row, kFullM);
        for (size_t col = 0; col < kFullN; ++col) {

            float acc = 0.0f;
            for (size_t kk = 0; kk < kFullK; kk += kK) {
                acc = MulVecVecHopperEmu(
                    acc,
                    a_mat[GetPos(row, kk, kFullK) / kK],
                    b_mat[GetPos(col, kk, kFullK) / kK]
                );
            }

            const auto our_res = __float2bfloat16(acc);
            const auto c_pos = GetPos(row, col, kFullN);
            const auto ref_res = c_mat[c_pos / kK][c_pos % kK];

            if (our_res != ref_res) {
                errx(
                    1,
                    "difference: %f vs. %f\n",
                    __bfloat162float(our_res),
                    __bfloat162float(ref_res)
                );
            }
        }
    }
}
