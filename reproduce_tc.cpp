#include "common.h"
#include "gemm_hopper_emu.h"

#include <cstdio>
#include <err.h>
#include <string>
#include <thread>
#include <vector>

#include <cuda_bf16.h>

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

int main(int argc, char** argv) {
    if (argc != 9) {
        errx(1, "usage: a_file b_file c_file m n k n_workers n_k_parts");
    }

    const size_t full_m = std::stoull(argv[4]);
    const size_t full_n = std::stoull(argv[5]);
    const size_t full_k = std::stoull(argv[6]);

    const auto a_mat = ParseFile(argv[1], full_m, full_k);
    const auto b_mat = ParseFile(argv[2], full_n, full_k);
    const auto c_mat = ParseFile(argv[3], full_m, full_n);

    const size_t n_workers = std::stoull(argv[7]);
    const size_t n_k_parts = std::stoull(argv[8]);

    std::vector<std::thread> workers;
    const auto chunk_size = full_m / n_workers;
    for (size_t w_idx = 0; w_idx < n_workers; ++w_idx) {
        workers.emplace_back([&, w_idx] {
            const auto start = w_idx * chunk_size;
            const auto end = (w_idx + 1) * chunk_size;
            for (size_t row = start; row < end; ++row) {
                printf("worker %zu: %zu/%zu\n", w_idx, row - start, chunk_size);
                for (size_t col = 0; col < full_n; ++col) {
                    float outer_acc = 0.0f;
                    const auto k_chunk_size = full_k / n_k_parts;

                    for (size_t part = 0; part < n_k_parts; ++part) {
                        float inner_acc = 0.0f;
                        const auto k_start = part * k_chunk_size;
                        const auto k_end = (part + 1) * k_chunk_size;
                        for (size_t kk = k_start; kk < k_end; kk += kK) {
                            inner_acc = MulVecVecHopperEmu(
                                inner_acc,
                                a_mat[GetPos(row, kk, full_k) / kK],
                                b_mat[GetPos(col, kk, full_k) / kK]
                            );
                        }
                        outer_acc += inner_acc;
                    }

                    const auto our_res = __float2bfloat16(outer_acc);
                    const auto c_pos = GetPos(row, col, full_n);
                    const auto ref_res = c_mat[c_pos / kK][c_pos % kK];

                    if (our_res != ref_res) {
                        errx(
                            1,
                            "difference at %zu, %zu: %f vs. %f\n",
                            row,
                            col,
                            __bfloat162float(our_res),
                            __bfloat162float(ref_res)
                        );
                    }
                }
            }
        });
    }

    for (auto& ww : workers) {
        ww.join();
    }
}
