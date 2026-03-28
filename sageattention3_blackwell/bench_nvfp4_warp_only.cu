#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 单warp纯计算微基准：
// - 只launch 1个warp（<<<1, 32>>>）
// - x与s在kernel内直接赋值
// - 循环repeat多次放大计算耗时
//
// 编译运行：
// nvcc bench_nvfp4_warp_only.cu -O3 -o bench_nvfp4_warp_only &&
// ./bench_nvfp4_warp_only

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err__));                                      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// __device__ __constant__ float kXiTable[8] = {0.0f,  -0.5f, -1.0f, -1.5f,
//                                              -2.0f, -3.0f, -4.0f, -6.0f};

__global__ void kernel_direct_warp(float *out, int repeat) {
  int lane = threadIdx.x & 31;
  if (threadIdx.x >= 32)
    return;

  constexpr float log2e = 1.4426950408889634f;
  float kXiTable[8] = {0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

  // 在kernel内直接赋值s和x(index)
  float s = 64.0f + 0.125f * static_cast<float>(lane);
  uint8_t idx = static_cast<uint8_t>(lane & 7);
  float xi = kXiTable[idx];

  float acc = 0.0f;
#pragma unroll 1
  for (int i = 0; i < repeat; ++i) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float xi = kXiTable[k];
      float y = ptx_exp2(xi * s * log2e);
      acc += y;
    }
  }
  out[lane] = acc;
}

__global__ void kernel_reuse_warp(float *out, int repeat) {
  int lane = threadIdx.x & 31;
  if (threadIdx.x >= 32)
    return;

  constexpr float log2e = 1.4426950408889634f;

  // 在kernel内直接赋值s和x(index)
  float s = 64.0f + 0.125f * static_cast<float>(lane);
  uint8_t idx = static_cast<uint8_t>(lane & 7);

  float acc = 0.0f;
#pragma unroll 1
  for (int i = 0; i < repeat; ++i) {
    float es = ptx_exp2(s * log2e);
    float inv_es = 1.0f / es;
    float inv_sqrt_es = rsqrtf(es);

    float inv_es2 = inv_es * inv_es;
    float inv_es3 = inv_es2 * inv_es;
    float inv_es4 = inv_es2 * inv_es2;
    float inv_es6 = inv_es4 * inv_es2;
    float inv_es_sqrt = inv_es * inv_sqrt_es;

    float lut[8];
    lut[0] = 1.0f;
    lut[1] = inv_sqrt_es;
    lut[2] = inv_es;
    lut[3] = inv_es_sqrt;
    lut[4] = inv_es2;
    lut[5] = inv_es3;
    lut[6] = inv_es4;
    lut[7] = inv_es6;

#pragma unroll
    for (int k = 0; k < 8; ++k) {
      acc += lut[k];
    }
  }
  out[lane] = acc;
}

float run_kernel_ms(void (*kernel)(float *, int), float *d_out, int repeat,
                    int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    kernel<<<1, 32>>>(d_out, repeat);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t st, ed;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&ed));

  CUDA_CHECK(cudaEventRecord(st));
  for (int i = 0; i < iters; ++i) {
    kernel<<<1, 32>>>(d_out, repeat);
  }
  CUDA_CHECK(cudaEventRecord(ed));
  CUDA_CHECK(cudaEventSynchronize(ed));
  CUDA_CHECK(cudaGetLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed));
  CUDA_CHECK(cudaEventDestroy(st));
  CUDA_CHECK(cudaEventDestroy(ed));
  return ms / static_cast<float>(iters);
}

int main(int argc, char **argv) {
  int repeat = 100000;
  int warmup = 20;
  int iters = 200;

  if (argc > 1)
    repeat = std::atoi(argv[1]);
  if (argc > 2)
    warmup = std::atoi(argv[2]);
  if (argc > 3)
    iters = std::atoi(argv[3]);

  if (repeat <= 0 || warmup < 0 || iters <= 0) {
    fprintf(stderr, "Invalid args: repeat>0, warmup>=0, iters>0\n");
    return 1;
  }

  printf("Single-warp microbench: repeat=%d, warmup=%d, iters=%d\n", repeat,
         warmup, iters);

  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * 32));

  float ms_direct =
      run_kernel_ms(kernel_direct_warp, d_out, repeat, warmup, iters);
  float ms_reuse =
      run_kernel_ms(kernel_reuse_warp, d_out, repeat, warmup, iters);

  float h_out_direct[32], h_out_reuse[32];

  CUDA_CHECK(cudaMemcpy(h_out_direct, d_out, sizeof(float) * 32,
                        cudaMemcpyDeviceToHost));

  ms_reuse = run_kernel_ms(kernel_reuse_warp, d_out, repeat, warmup, iters);
  CUDA_CHECK(cudaMemcpy(h_out_reuse, d_out, sizeof(float) * 32,
                        cudaMemcpyDeviceToHost));

  double max_abs = 0.0, mean_abs = 0.0;
  for (int i = 0; i < 32; ++i) {
    double a = h_out_direct[i];
    double b = h_out_reuse[i];
    double abs_err = fabs(a - b);
    if (abs_err > max_abs)
      max_abs = abs_err;
    mean_abs += abs_err;
  }
  mean_abs /= 32.0;

  printf("Direct (single warp): %.6f ms/launch\n", ms_direct);
  printf("Reuse  (single warp): %.6f ms/launch\n", ms_reuse);
  printf("Speedup direct/reuse: %.4fx\n", ms_direct / ms_reuse);
  printf("Abs error: max_abs=%.6e, mean_abs=%.6e\n", max_abs, mean_abs);

  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
