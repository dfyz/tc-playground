#include <cuda.h>
#include <err.h>

void CheckCu(CUresult res, const char* reason) {
    if (res != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(res, &err); // ignore errors here
        errx(1, "failed to %s: %s", reason, err);
    }
}

int main() {
    CheckCu(cuInit(0), "initialize CUDA");

    CUdevice device;
    CheckCu(cuDeviceGet(&device, 0), "get CUDA device #0");

    CUcontext ctx;
    CheckCu(cuCtxCreate(&ctx, nullptr, 0, device), "get CUDA context");

    CUmodule module;
    CheckCu(cuModuleLoad(&module, "tc.cubin"), "load the compiled PTX");

    CUfunction run_tc;
    CheckCu(cuModuleGetFunction(&run_tc, module, "run_tc"), "find the run_tc kernel");

    void* args[] = {
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
            args,
            nullptr   // nothing extra
        ),
        "launch the run_tc kernel"
    );

    CheckCu(cuCtxSynchronize(), "wait for run_tc completion");

    CheckCu(cuModuleUnload(module), "unload the compiled PTX");
    CheckCu(cuCtxDestroy(ctx), "tear down the CUDA context");
}
