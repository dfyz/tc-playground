#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <err.h>
#include <iostream>
#include <random>
#include <utility>

void CheckCu(CUresult res, const char* reason) {
    if (res != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(res, &err); // ignore errors here
        errx(1, "failed to %s: %s", reason, err);
    }
}

using Rng = std::mt19937_64;
constexpr Rng::result_type kSeed = 31337;

using Vec16 = std::array<__nv_bfloat16, 16>;
using MatA  = std::array<Vec16, 64>;
using MatB  = std::array<Vec16, 256>;

void GenVec(Rng& rng, Vec16& out) {
    std::uniform_int_distribution<uint16_t> gen;
    for (size_t ii = 0; ii < out.size(); ++ii) {
        out[ii] = __nv_bfloat16_raw{gen(rng)};
    }
}

void PermuteVec(Rng& rng, Vec16& out) {
    for (size_t ii = 1; ii < out.size(); ++ii) {
        std::uniform_int_distribution<size_t> gen{0, ii};
        const size_t jj = gen(rng);
        std::swap(out[ii], out[jj]);
    }
}

std::pair<MatA, MatB> GenInput() {
    Rng rng{kSeed};
    MatA res_a;
    MatB res_b;

    // First 64 vectors of both matrices are permutations of the first vector.
    GenVec(rng, res_a[0]);
    GenVec(rng, res_b[0]);
    for (size_t ii = 1; ii < res_a.size(); ++ii) {
        res_a[ii] = res_a[0];
        res_b[ii] = res_b[0];

        PermuteVec(rng, res_a[ii]);
        PermuteVec(rng, res_b[ii]);
    }

    // The remaining vectors of B are just random vectors.
    for (size_t ii = res_a.size(); ii < res_b.size(); ++ii) {
        GenVec(rng, res_b[ii]);
    }

    return std::make_pair(res_a, res_b);
}

int main() {
    const auto [mat_a, mat_b] = GenInput();

    CheckCu(cuInit(0), "initialize CUDA");

    CUdevice device;
    CheckCu(cuDeviceGet(&device, 0), "get CUDA device #0");

    CUcontext ctx;
    CheckCu(cuCtxCreate(&ctx, nullptr, 0, device), "get CUDA context");

    CUmodule module;
    CheckCu(cuModuleLoad(&module, "tc.cubin"), "load the compiled PTX");

    CUfunction run_tc;
    CheckCu(cuModuleGetFunction(&run_tc, module, "run_tc"), "find the run_tc kernel");

    std::array<void*, 3> args = {
        // FIXME: use real data
        nullptr, // A matrix
        nullptr, // B matrix
        nullptr, // output matrix
    };
    CheckCu(
        cuLaunchKernel(
            run_tc,
            1, 1, 1,   // grid dims
            128, 1, 1, // block dims
            0,         // don't need dynamic smem
            nullptr,   // use the default stream
            args.data(),
            nullptr   // nothing extra
        ),
        "launch the run_tc kernel"
    );

    CheckCu(cuCtxSynchronize(), "wait for run_tc completion");

    CheckCu(cuModuleUnload(module), "unload the compiled PTX");
    CheckCu(cuCtxDestroy(ctx), "tear down the CUDA context");
}
