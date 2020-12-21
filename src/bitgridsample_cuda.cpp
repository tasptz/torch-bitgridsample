#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void runKernel(
    const unsigned char *in, const uint32_t batch, const uint32_t widthIn, const uint32_t heightIn,
    const float *grid,
    float *out, uint32_t const outWidth, uint32_t const outHeight, uint32_t const channels,
    float *outGrad,
    cudaStream_t stream
);

void forward(const at::Tensor in, const at::Tensor grid, at::Tensor out, at::Tensor outGrad) {
    const auto batch = in.size(0);
    const auto heightIn = in.size(1);
    const auto widthIn = in.size(2);

    const auto heightOut = out.size(1);
    const auto widthOut = out.size(2);
    const auto channels = out.size(3);

    runKernel(
        in.data_ptr<unsigned char>(), batch, widthIn, heightIn,
        grid.data_ptr<float>(),
        out.data_ptr<float>(), widthOut, heightOut, channels,
        outGrad.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
}