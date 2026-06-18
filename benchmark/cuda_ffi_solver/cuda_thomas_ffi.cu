#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kScratchArrays = 6;
constexpr int kMaxFibersPerBlock = 8;
constexpr int64_t kSoftSharedMemoryBytes = 40 * 1024;

__global__ void DoubleCableThomasKernel(
    const float* a00,
    const float* a01,
    const float* a10,
    const float* a11,
    const float* off0,
    const float* off1,
    const float* rhs0,
    const float* rhs1,
    float* out0,
    float* out1,
    int64_t batch_size,
    int64_t nx) {
  int lane = threadIdx.x;
  int64_t batch = static_cast<int64_t>(blockIdx.x) * blockDim.x + lane;
  if (batch >= batch_size) {
    return;
  }

  extern __shared__ float scratch[];
  int64_t lane_base = static_cast<int64_t>(lane) * nx;
  int64_t block_stride = static_cast<int64_t>(blockDim.x) * nx;
  float* c00 = scratch + 0 * block_stride + lane_base;
  float* c01 = scratch + 1 * block_stride + lane_base;
  float* c10 = scratch + 2 * block_stride + lane_base;
  float* c11 = scratch + 3 * block_stride + lane_base;
  float* d0 = scratch + 4 * block_stride + lane_base;
  float* d1 = scratch + 5 * block_stride + lane_base;

  int64_t base = batch * nx;
  int64_t edge_base = batch * (nx - 1);

  float m00 = a00[base];
  float m01 = a01[base];
  float m10 = a10[base];
  float m11 = a11[base];
  float r0 = rhs0[base];
  float r1 = rhs1[base];
  float det = m00 * m11 - m01 * m10;
  float inv00 = m11 / det;
  float inv01 = -m01 / det;
  float inv10 = -m10 / det;
  float inv11 = m00 / det;

  float u0 = off0[edge_base];
  float u1 = off1[edge_base];
  float prev_c00 = inv00 * u0;
  float prev_c01 = inv01 * u1;
  float prev_c10 = inv10 * u0;
  float prev_c11 = inv11 * u1;
  float prev_d0 = inv00 * r0 + inv01 * r1;
  float prev_d1 = inv10 * r0 + inv11 * r1;
  c00[0] = prev_c00;
  c01[0] = prev_c01;
  c10[0] = prev_c10;
  c11[0] = prev_c11;
  d0[0] = prev_d0;
  d1[0] = prev_d1;

  for (int64_t i = 1; i < nx; ++i) {
    int64_t offset = base + i;
    float l0 = off0[edge_base + i - 1];
    float l1 = off1[edge_base + i - 1];
    m00 = a00[offset] - l0 * prev_c00;
    m01 = a01[offset] - l0 * prev_c01;
    m10 = a10[offset] - l1 * prev_c10;
    m11 = a11[offset] - l1 * prev_c11;
    r0 = rhs0[offset] - l0 * prev_d0;
    r1 = rhs1[offset] - l1 * prev_d1;

    det = m00 * m11 - m01 * m10;
    inv00 = m11 / det;
    inv01 = -m01 / det;
    inv10 = -m10 / det;
    inv11 = m00 / det;

    u0 = 0.0f;
    u1 = 0.0f;
    if (i < nx - 1) {
      u0 = off0[edge_base + i];
      u1 = off1[edge_base + i];
    }
    prev_c00 = inv00 * u0;
    prev_c01 = inv01 * u1;
    prev_c10 = inv10 * u0;
    prev_c11 = inv11 * u1;
    prev_d0 = inv00 * r0 + inv01 * r1;
    prev_d1 = inv10 * r0 + inv11 * r1;
    c00[i] = prev_c00;
    c01[i] = prev_c01;
    c10[i] = prev_c10;
    c11[i] = prev_c11;
    d0[i] = prev_d0;
    d1[i] = prev_d1;
  }

  float x0 = d0[nx - 1];
  float x1 = d1[nx - 1];
  out0[base + nx - 1] = x0;
  out1[base + nx - 1] = x1;

  for (int64_t rev = 0; rev < nx - 1; ++rev) {
    int64_t i = nx - 2 - rev;
    float next_x0 = x0;
    float next_x1 = x1;
    x0 = d0[i] - c00[i] * next_x0 - c01[i] * next_x1;
    x1 = d1[i] - c10[i] * next_x0 - c11[i] * next_x1;
    out0[base + i] = x0;
    out1[base + i] = x1;
  }
}

std::string ShapeString(ffi::BufferR2<ffi::F32> buffer) {
  auto dims = buffer.dimensions();
  std::ostringstream stream;
  stream << "(" << dims[0] << ", " << dims[1] << ")";
  return stream.str();
}

ffi::Error CheckShape(
    const char* name,
    ffi::BufferR2<ffi::F32> buffer,
    int64_t expected0,
    int64_t expected1) {
  auto dims = buffer.dimensions();
  if (dims[0] != expected0 || dims[1] != expected1) {
    std::ostringstream stream;
    stream << name << " must have shape (" << expected0 << ", " << expected1
           << "), got " << ShapeString(buffer) << ".";
    return ffi::Error::InvalidArgument(stream.str());
  }
  return ffi::Error::Success();
}

int ChooseFibersPerBlock(int64_t nx) {
  int64_t bytes_per_fiber = kScratchArrays * nx * static_cast<int64_t>(sizeof(float));
  if (bytes_per_fiber <= 0) {
    return 0;
  }
  int64_t fit = kSoftSharedMemoryBytes / bytes_per_fiber;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(kMaxFibersPerBlock, fit)));
}

ffi::Error DoubleCableThomasImpl(
    cudaStream_t stream,
    ffi::BufferR2<ffi::F32> a00,
    ffi::BufferR2<ffi::F32> a01,
    ffi::BufferR2<ffi::F32> a10,
    ffi::BufferR2<ffi::F32> a11,
    ffi::BufferR2<ffi::F32> off0,
    ffi::BufferR2<ffi::F32> off1,
    ffi::BufferR2<ffi::F32> rhs0,
    ffi::BufferR2<ffi::F32> rhs1,
    ffi::ResultBufferR2<ffi::F32> out0,
    ffi::ResultBufferR2<ffi::F32> out1) {
  auto rhs_dims = rhs0.dimensions();
  int64_t batch_size = rhs_dims[0];
  int64_t nx = rhs_dims[1];
  if (batch_size < 1) {
    return ffi::Error::InvalidArgument("batch_size must be >= 1.");
  }
  if (nx < 2) {
    return ffi::Error::InvalidArgument("Nx must be >= 2.");
  }

  for (auto error : {
           CheckShape("a00", a00, batch_size, nx),
           CheckShape("a01", a01, batch_size, nx),
           CheckShape("a10", a10, batch_size, nx),
           CheckShape("a11", a11, batch_size, nx),
           CheckShape("off0", off0, batch_size, nx - 1),
           CheckShape("off1", off1, batch_size, nx - 1),
           CheckShape("rhs1", rhs1, batch_size, nx),
           CheckShape("out0", *out0, batch_size, nx),
           CheckShape("out1", *out1, batch_size, nx),
       }) {
    if (error.failure()) {
      return error;
    }
  }

  int fibers_per_block = ChooseFibersPerBlock(nx);
  int64_t shared_bytes = static_cast<int64_t>(fibers_per_block) * kScratchArrays * nx *
                         static_cast<int64_t>(sizeof(float));
  if (shared_bytes > 48 * 1024) {
    std::ostringstream stream_msg;
    stream_msg << "Nx=" << nx
               << " requires too much dynamic shared memory for the first CUDA FFI spike.";
    return ffi::Error::InvalidArgument(stream_msg.str());
  }

  int64_t blocks = (batch_size + fibers_per_block - 1) / fibers_per_block;
  DoubleCableThomasKernel<<<static_cast<unsigned int>(blocks), fibers_per_block,
                            static_cast<size_t>(shared_bytes), stream>>>(
      a00.typed_data(),
      a01.typed_data(),
      a10.typed_data(),
      a11.typed_data(),
      off0.typed_data(),
      off1.typed_data(),
      rhs0.typed_data(),
      rhs1.typed_data(),
      out0->typed_data(),
      out1->typed_data(),
      batch_size,
      nx);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return ffi::Error::Internal(cudaGetErrorString(error));
  }
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AxonScopeDoubleCableThomasF32,
    DoubleCableThomasImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR2<ffi::F32>>()  // a00
        .Arg<ffi::BufferR2<ffi::F32>>()  // a01
        .Arg<ffi::BufferR2<ffi::F32>>()  // a10
        .Arg<ffi::BufferR2<ffi::F32>>()  // a11
        .Arg<ffi::BufferR2<ffi::F32>>()  // off0
        .Arg<ffi::BufferR2<ffi::F32>>()  // off1
        .Arg<ffi::BufferR2<ffi::F32>>()  // rhs0
        .Arg<ffi::BufferR2<ffi::F32>>()  // rhs1
        .Ret<ffi::BufferR2<ffi::F32>>()  // out0
        .Ret<ffi::BufferR2<ffi::F32>>()  // out1
);

