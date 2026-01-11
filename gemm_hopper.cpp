#include "gemm_hopper.h"

#include <cmath>
#include <cstdio>
#include <err.h>

#include <cuda.h>

namespace {

void CheckCu(CUresult res, const char* reason) {
    if (res != CUDA_SUCCESS) {
        const char* err;
        const auto err_res = cuGetErrorString(res, &err);
        errx(1, "failed to %s: %s", reason, err_res == CUDA_SUCCESS ? err : "N/A");
    }
}

CUdeviceptr GetDevicePointer(void* host_pointer) {
    CUdeviceptr res;
    CheckCu(cuMemHostGetDevicePointer(&res, host_pointer, 0), "get the device pointer");
    return res;
}

class CuContext {
public:
    CuContext() {
        CheckCu(cuDeviceGet(&device_, 0), "get CUDA device #0");
        CheckCu(cuCtxCreate(&ctx_, nullptr, 0, device_), "get CUDA context");
    }

    ~CuContext() {
        CheckCu(cuCtxDestroy(ctx_), "tear down the CUDA context");
    }

private:
    CUdevice device_;
    CUcontext ctx_;
};

class SmemInput {
public:
    SmemInput(size_t rows)
        // Without swizzling, we skip exactly `rows` 128-bit elements.
        : lbo_(rows * 16)
    {
        CheckCu(cuMemHostAlloc(
            (void**)&data_, rows * kK * sizeof(__nv_bfloat16), CU_MEMHOSTALLOC_DEVICEMAP),
            "allocate shared memory input"
        );
    }

    ~SmemInput() {
        CheckCu(cuMemFreeHost(data_), "free shared memory input");
    }

    __nv_bfloat16& operator()(size_t row, size_t col) {
        const size_t off =
            row * 8 +
            (col / 8) * (lbo_ / sizeof(__nv_bfloat16)) +
            (col % 8);
        ;
        return *(data_ + off);
    }

    template <size_t Rows>
    void CopyFrom(const std::array<Vec, Rows>& mat) {
        for (size_t row = 0; row < mat.size(); ++row) {
            const auto& vec = mat[row];
            for (size_t col = 0; col < vec.size(); ++col) {
                (*this)(row, col) = vec[col];
            }
        }
    }

    CUdeviceptr device_data() const {
        return GetDevicePointer(data_);
    }

    uint64_t desc() const {
        return ((lbo_ >> 4) << 16) | ((sbo_ >> 4) << 32);
    }

private:
    // 9.7.15.5.1.2.1.3: "T = 128 / sizeof-elements-in-bits [...]
    // represents scale factor which normalizes matrix element types to 128-bits."
    const size_t t_ = 128 / sizeof(__nv_bfloat16);
    // 9.7.15.5.1.9:
    // "The offset from the first 8 rows to the next 8 rows"
    // Without swizzling, it's always 128 bytes (8 128-bit elements).
    const size_t sbo_ = 128;
    // 9.7.15.5.1.8: "the offset from the first column
    // to the second columns [sic] of the 8x2 tile
    // in the 128-bit element type normalized matrix"
    const size_t lbo_;

    __nv_bfloat16* data_;
};

class GmemOutput {
public:
    GmemOutput(size_t rows, size_t cols)
        : rows_(rows)
        , cols_(cols)
    {
        CheckCu(
            cuMemHostAlloc((void**)&data_, rows * cols * sizeof(float), CU_MEMHOSTALLOC_DEVICEMAP),
            "allocate global memory output"
        );
    }

    ~GmemOutput() {
        CheckCu(cuMemFreeHost(data_), "free global memory output");
    }

    float operator()(size_t row, size_t col) const {
        // 9.7.15.5.1.1.1, "Accumulator D" with `.f32`
        const size_t tid =
            32 * (row / 16) +
            4 * (row % 8) +
            (col % 8) / 2
        ;

        const size_t off =
            4 * (col / 8) +
            2 * ((row / 8) % 2) +
            col % 2
        ;

        return data_[(kN / 2) * tid + off];
    }

    CUdeviceptr device_data() const {
        return GetDevicePointer(data_);
    }

private:
    size_t rows_;
    size_t cols_;
    float* data_;
};

}

GemmOutput MulMatMatHopper(const MatA& mat_a, const MatB& mat_b) {
    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to initialize CUDA; device results will be unavailiable\n");
        GemmOutput dummy_result;
        for (auto& row : dummy_result) {
            row.fill(std::nanf("NA"));
        }
        return dummy_result;
    }

    CuContext _ctx;

    CUmodule module;
    CheckCu(cuModuleLoad(&module, "tc.cubin"), "load the compiled PTX");

    CUfunction run_tc;
    CheckCu(cuModuleGetFunction(&run_tc, module, "run_tc"), "find the run_tc kernel");

    SmemInput smem_a(kM);
    SmemInput smem_b(kN);

    smem_a.CopyFrom(mat_a);
    smem_b.CopyFrom(mat_b);

    GmemOutput device_out{kM, kN};

    auto a_device_data = smem_a.device_data();
    auto b_device_data = smem_b.device_data();
    auto a_desc = smem_a.desc();
    auto b_desc = smem_b.desc();
    auto out_device_data = device_out.device_data();

    std::array<void*, 5> args = {
        &a_device_data, &b_device_data,
        &a_desc, &b_desc,
        &out_device_data,
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

    GemmOutput result;
    for (size_t row = 0; row < mat_a.size(); ++row) {
        for (size_t col = 0; col < mat_b.size(); ++col) {
            result[row][col] = device_out(row, col);
        }
    }
    return result;
}
