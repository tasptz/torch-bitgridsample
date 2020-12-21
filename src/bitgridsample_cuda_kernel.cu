#include <stdint.h>
#include <stdexcept>

__forceinline__ __device__ void extract(const unsigned char *p,
    const int x, const int y,
    const int width, const int height, const int channels,
    float * const out) {

    const bool pad = x < 0 || y < 0 || x >= width || y >= height;
    const unsigned char v = pad ? 0 : p[y * width + x];
    for (int i = 0; i < channels; ++i) {
        out[i] = static_cast<float>((v >> i) & 1);
    }
}

/**
 * struct ApplyGridSample<scalar_t, 2, GridSamplerInterpolation::Bilinear, padding, align_corners>
 * https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/GridSamplerKernel.cpp#L453
 *
 * inline void backward(TensorAccessor<scalar_t, 3>& gInp_slice,
 *                      TensorAccessor<scalar_t, 3>& gGrid_slice,
 *                      const TensorAccessor<scalar_t, 3>& gOut_slice,
 *                      const TensorAccessor<scalar_t, 3>& inp_slice,
 *                      int64_t offset, const Vec& grid_x, const Vec& grid_y,
 *                      int64_t len) const {
 * https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/GridSamplerKernel.cpp#L583
 */
__device__ void sample(
    const unsigned char *in, const float fx, const float fy, const int width, const int height,
    float *out, const int channels,
    float *outGrad) {

    // north-west    north-east
    // nw------------ne
    // |    n        |
    // | w  .(x,y) e |
    // |    s        |
    // sw------------se
    // south-west    south-east
    const float flx = floor(fx);
    const float fly = floor(fy);
    const float w = fx - flx;
    const float e = 1.f - w;
    const float n = fy - fly;
    const float s = 1.f - n;
    const int ix = static_cast<int>(flx);
    const int iy = static_cast<int>(fly);
    float nw[8], ne[8], se[8], sw[8];
    extract(in, ix,     iy,     width, height, channels, nw);
    extract(in, ix + 1, iy,     width, height, channels, ne);
    extract(in, ix + 1, iy + 1, width, height, channels, se);
    extract(in, ix    , iy + 1, width, height, channels, sw);
    for (int i = 0; i < channels; ++i) {
        out[i] = (nw[i] * e + ne[i] * w) * s + (sw[i] * e + se[i] * w) * n;
        outGrad[i * 2] =     ((ne[i] - nw[i]) * s + (se[i] - sw[i]) * n) * width / 2.f;
        outGrad[i * 2 + 1] = ((sw[i] - nw[i]) * e + (se[i] - ne[i]) * w) * height / 2.f;
    }
}

__global__ void kernel(
    const unsigned char *in, const uint32_t batch, const uint32_t widthIn, const uint32_t heightIn,
    const float *grid,
    float *out, const uint32_t channels,
    float *outGrad) {

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int widthOut = gridDim.x * blockDim.x;
    const int heightOut = gridDim.y * blockDim.y;

    for (int b = 0; b < batch; ++b) {
        const int idxGrid = (b * heightIn * widthIn * 2) + (y * widthIn + x) * 2;
        const float fx = grid[idxGrid] * static_cast<float>(widthIn) / 2.f + static_cast<float>(widthIn - 1) / 2.f;
        const float fy = grid[idxGrid + 1] * static_cast<float>(heightIn) / 2.f + static_cast<float>(heightIn - 1) / 2.f;

        sample(in + (b * heightIn * widthIn),
            fx, fy, widthIn, heightIn,
            out + (b * heightOut * widthOut * channels) + (y * widthOut + x) * channels, channels,
            outGrad + (b * heightOut * widthOut * channels * 2) + (y * widthOut + x) * channels * 2);
    }
}

void runKernel(
    const unsigned char *in, const uint32_t batch, const uint32_t widthIn, uint32_t const heightIn,
    const float *grid,
    float *out, const uint32_t widthOut, const uint32_t heightOut, const uint32_t channels,
    float *outGrad,
    cudaStream_t stream) {

    if (heightOut < BLOCKSIZE || heightOut % BLOCKSIZE != 0 || widthOut < BLOCKSIZE || widthOut % BLOCKSIZE != 0)
        throw std::runtime_error("Dimensions do not match cuda block size");

    dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridDim(widthOut / BLOCKSIZE, heightOut / BLOCKSIZE, 1);

    kernel<<<blockDim, gridDim, 0, stream>>>(in, batch, widthIn, heightIn, grid, out, channels, outGrad);
}
