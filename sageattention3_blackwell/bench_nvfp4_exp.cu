#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

// nvcc -lineinfo bench_nvfp4_exp.cu -o bench_nvfp4_exp
// ncu --set full -o my_profile ./bench_nvfp4_exp

/*
nvcc bench_nvfp4_exp.cu -diag-suppress 177 -O3 -std=c++17 -arch=sm_120a -o
bench_nvfp4_exp
&&
./bench_nvfp4_exp
*/
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err__));                                      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ---------------- PTX exp2 ----------------
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// x_i 取值集合: {0,-0.5,-1,-1.5,-2,-3,-4,-6}
// 用 index 0..7 编码，避免直接存 half。//这个是个好方法，不用再计算索引了
__device__ __constant__ float kXiTable[8] = {0.0f,  -0.5f, -1.0f, -1.5f,
                                             -2.0f, -3.0f, -4.0f, -6.0f};

// -------- 方案A：直接算 e^(x_i * s) --------
__global__ void kernel_direct(const float *__restrict__ s_arr,
                              const uint8_t *__restrict__ xi_idx,
                              float *__restrict__ out, int n_s, int ratio) {
  //   constexpr int Element_per_thread = 32;
  int S_PER_THREAD = 1; // 每线程处理1个s，对应32个元素
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int s_base = tid * S_PER_THREAD;
  if (s_base >= n_s)
    return;

  constexpr float log2e = 1.4426950408889634f;

#pragma unroll
  for (int j = 0; j < S_PER_THREAD; ++j) {
    int sid = s_base + j;
    if (sid >= n_s)
      break;

    float s = s_arr[sid];
    int base = sid * ratio;

#pragma unroll
    for (int k = 0; k < ratio; ++k) {
      int gid = base + k;
      float xi = kXiTable[xi_idx[gid] & 7];
      // 这里似乎有点对direct不公平，因为如果direct的话，输入就是xi*s了，会少一次乘法
      out[gid] = ptx_exp2(xi * s * log2e);
    }
  }
}

// -------- 方案B：先算 es，再用平方/sqrt得到 e^(x_i*s) --------
// 由于 x_i 仅来自离散集合，可通过 es 的幂关系得到：
// x=0      -> 1
// x=-0.5   -> 1/sqrt(es)
// x=-1     -> 1/es
// x=-1.5   -> 1/(es*sqrt(es))
// x=-2     -> 1/es^2
// x=-3     -> 1/es^3
// x=-4     -> 1/es^4
// x=-6     -> 1/es^6
__global__ void kernel_reuse_es(const float *__restrict__ s_arr,
                                const uint8_t *__restrict__ xi_idx,
                                float *__restrict__ out, int n_s, int ratio) {
  constexpr int S_PER_THREAD = 1; // 每线程处理1个s，对应32个元素
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int s_base = tid * S_PER_THREAD;
  if (s_base >= n_s)
    return;

  constexpr float log2e = 1.4426950408889634f;

  // #pragma unroll
  for (int j = 0; j < S_PER_THREAD; ++j) {
    int sid = s_base + j;
    if (sid >= n_s)
      break;

    float s = s_arr[sid];
    int base = sid * ratio;

    float es = ptx_exp2(s * log2e); // e^s
    float inv_es = 1.0f / es;
    float inv_sqrt_es = rsqrtf(es);

    // 固定计算 + 表查
    float inv_es2 = inv_es * inv_es;          // e^-2s
    float inv_es3 = inv_es2 * inv_es;         // e^-3s
    float inv_es4 = inv_es2 * inv_es2;        // e^-4s
    float inv_es6 = inv_es4 * inv_es2;        // e^-6s
    float inv_es_sqrt = inv_es * inv_sqrt_es; // e^-1.5s

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
    for (int k = 0; k < 32; ++k) {
      int gid = base + k;
      uint8_t idx = xi_idx[gid] & 7;
      out[gid] = lut[idx];
    }
  }
}

__global__ void bench_pure_compute_cycles(float *out_dummy) {
  // 1. 模拟 Fragment，强制分配到寄存器
  float s = 2.456f;
  float dependency = 0.0f; // 用于制造数据依赖，防止被编译器优化掉
  float xi_vals[32] = {0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
                       0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
                       0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
                       0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  float frags[32];
  constexpr float log2e = 1.4426950408889634f;

  // 预热时钟
  long long start = clock64();

// 2. 强制展开并执行多次，放大差异，同时制造数据依赖防止被编译器优化(DCE)
#pragma unroll 1
  // 这里不应该unroll,unroll后面的数字表示展开份数，1即不展开
  // 把iter改变，几乎完美的线性，也证明确实没有展开
  for (int iter = 0; iter < 1000; ++iter) {

    // --- 测试方案 A (Direct) ---
    // k8:129240
    // exp单元远远没有被塞满，后面多的exp很好的被隐藏了，测了一下一直到k14增长都很小，k15跳的多了一下
    // k16:138233
    // 可能在这里到达了exp单元的吞吐极限。所以其实对于直接的计算一个线程分16个exp实际上比较好?
    // k32:220198 cycles

    // #pragma unroll
    //     for (int k = 0; k < 8; ++k) {
    //       frags[k] = ptx_exp2(xi_vals[k] * s * log2e);
    //       // 制造数据依赖，如果删掉，cycle打印出来只有1，可能直接全部删掉了
    //       // 这里如果用s，本来可以unroll的就不能unroll了，因为s的依赖
    //       dependency += 0.00001f * frags[k];
    //     }
    //     s += dependency;

    // 加了上面这一句之后从41017变成129240
    // 感觉可能是不加这个的话，计算frags那一行实际上每一次结果都是一样的，得让他动起来

    // --- 测试方案 B (Reuse) 注释掉A测B ---
    // 222406

    float es = ptx_exp2(s * log2e);

    float inv_es = 1.0f / es;
    float inv_sqrt_es = rsqrtf(es);
    float inv_es2 = inv_es * inv_es;  // e^-2s
    float inv_es3 = inv_es2 * inv_es; // e^-3s
    // e^-4s都用一个寄存器做乘法会不会更慢，直觉上要先复制一个才能乘
    // 但是实际上比3*1更快，可能是因为数据依赖的距离
    float inv_es4 = inv_es2 * inv_es2;        // e^-4s
    float inv_es6 = inv_es3 * inv_es3;        // e^-6s
    float inv_es_sqrt = inv_es * inv_sqrt_es; // e^-1.5s

    frags[0] = 1.0f;
    frags[1] = inv_sqrt_es;
    frags[2] = inv_es;
    frags[3] = inv_es_sqrt;
    frags[4] = inv_es2;
    frags[5] = inv_es3;
    frags[6] = inv_es4;
    frags[7] = inv_es6;
    // frags[7] = 1.0f;

    // 同样制造依赖
    dependency += 0.00001f * frags[7];

    // 我把这个7改成iter%8竟然就能让cycle下降30%
    // 似乎是因为编译器直接把其他所有的frags计算消除掉了
    // 但是这样的话为什么单独写一个7不会被消除呢？
    // #pragma unroll
    //     for (int k = 0; k < 8; ++k) {
    //       s += 0.00001f * frags[k]; // 这里对s加，模拟移动寄存器的操作
    //     }
    s += 0.00001f * frags[0];
    s += 0.00001f * frags[1];
    s += 0.00001f * frags[2];
    s += 0.00001f * frags[3];
    s += 0.00001f * frags[4];
    s += 0.00001f * frags[5];
    s += 0.00001f * frags[6];
    s += 0.00001f * frags[7];

    // s += 0.00001f * frags[0];
    // s += 0.00001f * frags[1];
    // s += 0.00001f * frags[2];
    // s += 0.00001f * frags[3];
    // s += 0.00001f * frags[4];
    // s += 0.00001f * frags[5];
    // s += 0.00001f * frags[6];
    // s += 0.00001f * frags[7];

    dependency += s;
  }

  long long stop = clock64();

  // 3. 必须把结果写回到全局内存，否则整个计算会被编译器直接删掉！
  if (threadIdx.x == 0) {
    out_dummy[blockIdx.x] = dependency + (stop - start); // 把周期数也带出来看看
    printf("Elapsed cycles: %lld\n", stop - start);
  }
}

__global__ void bench_pure_compute_cycles2(float *out_dummy, float dynamic_s) {
  // 1. 模拟 Fragment，强制分配到寄存器
  float s = 2.456f;
  s = dynamic_s;
  float dependency = 0.0f; // 用于制造数据依赖，防止被编译器优化掉
  float xi_vals[32] = {0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
                       0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
                       0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
                       0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  float frags[32];
  constexpr float log2e = 1.4426950408889634f;

  // 预热时钟
  long long start = clock64();

  float es = ptx_exp2(s * log2e);
  //     dependency = fmaf(0.00001f, es, dependency);
  // s = fmaf(0.00001f, es, s);
  //   dependency += 0.00001f * es;
  //   s += 0.00001f * dependency; // 40019
  //   dependency += 0.00001f * es; // 43008
  //   dependency += 0.00001f * es; // 44014

  long long stop = clock64();
  if (stop > 10000000) {
    dependency += 0.00001f * es;
    dependency += (stop - start);
  }
  if (threadIdx.x == 0) {
    out_dummy[blockIdx.x] = dependency + (stop - start); // 把周期数也带出来看看
    printf("Elapsed cycles: %lld\n", stop - start);
  }
}

// 计时函数
float run_kernel_ms(void (*kernel)(const float *, const uint8_t *, float *, int,
                                   int),
                    const float *d_s, const uint8_t *d_xi, float *d_out,
                    int n_s, int ratio, int warmup, int iters) {
  constexpr int S_PER_THREAD = 1;
  int block = 256;
  int threads_needed = (n_s + S_PER_THREAD - 1) / S_PER_THREAD;
  int grid = (threads_needed + block - 1) / block;

  // warmup
  for (int i = 0; i < warmup; ++i) {
    kernel<<<grid, block>>>(d_s, d_xi, d_out, n_s, ratio);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t st, ed;
  CUDA_CHECK(cudaEventCreate(&st));
  CUDA_CHECK(cudaEventCreate(&ed));

  CUDA_CHECK(cudaEventRecord(st));
  for (int i = 0; i < iters; ++i) {
    kernel<<<grid, block>>>(d_s, d_xi, d_out, n_s, ratio);
  }
  CUDA_CHECK(cudaEventRecord(ed));
  CUDA_CHECK(cudaEventSynchronize(ed));
  CUDA_CHECK(cudaGetLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, st, ed));
  CUDA_CHECK(cudaEventDestroy(st));
  CUDA_CHECK(cudaEventDestroy(ed));
  return ms / iters;
}

int main(int argc, char **argv) {
  //   // 默认参数：可调
  //   int n_s = 1 << 22; // s的数量
  //   int ratio = 32;    // 每个s对应的xi数量（改为32）
  //   int warmup = 20;
  //   int iters = 200;

  //   if (argc > 1)
  //     n_s = std::atoi(argv[1]);
  //   if (argc > 2)
  //     ratio = std::atoi(argv[2]);
  //   if (argc > 3)
  //     warmup = std::atoi(argv[3]);
  //   if (argc > 4)
  //     iters = std::atoi(argv[4]);

  //   if (ratio <= 0) {
  //     fprintf(stderr, "ratio must be > 0\n");
  //     return 1;
  //   }

  //   int total = n_s * ratio;
  //   printf("n_s=%d, ratio=%d, total=%d, warmup=%d, iters=%d\n", n_s, ratio,
  //   total,
  //          warmup, iters);

  //   // 生成输入：s in [0, 448], xi index in [0,7]
  //   std::vector<float> h_s(n_s);
  //   std::vector<uint8_t> h_xi(total);
  //   std::vector<float> h_out_direct(total), h_out_reuse(total);

  //   std::mt19937 rng(12345);
  //   std::uniform_real_distribution<float> dist_s(0.0f, 448.0f);
  //   std::uniform_int_distribution<int> dist_xi(0, 7);

  //   for (int i = 0; i < n_s; ++i)
  //     h_s[i] = dist_s(rng);
  //   for (int i = 0; i < total; ++i)
  //     h_xi[i] = static_cast<uint8_t>(dist_xi(rng));

  //   float *d_s = nullptr, *d_out = nullptr;
  //   uint8_t *d_xi = nullptr;
  //   CUDA_CHECK(cudaMalloc(&d_s, sizeof(float) * n_s));
  //   CUDA_CHECK(cudaMalloc(&d_xi, sizeof(uint8_t) * total));
  //   CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * total));
  //   CUDA_CHECK(
  //       cudaMemcpy(d_s, h_s.data(), sizeof(float) * n_s,
  //       cudaMemcpyHostToDevice));
  //   CUDA_CHECK(cudaMemcpy(d_xi, h_xi.data(), sizeof(uint8_t) * total,
  //                         cudaMemcpyHostToDevice));

  //   // 计时：direct
  //   float ms_direct =
  //       run_kernel_ms(kernel_direct, d_s, d_xi, d_out, n_s, ratio, warmup,
  //       iters);
  //   CUDA_CHECK(cudaMemcpy(h_out_direct.data(), d_out, sizeof(float) * total,
  //                         cudaMemcpyDeviceToHost));

  //   // 计时：reuse_es
  //   float ms_reuse = run_kernel_ms(kernel_reuse_es, d_s, d_xi, d_out, n_s,
  //   ratio,
  //                                  warmup, iters);
  //   CUDA_CHECK(cudaMemcpy(h_out_reuse.data(), d_out, sizeof(float) * total,
  //                         cudaMemcpyDeviceToHost));

  //   // 简单误差统计
  //   double max_abs = 0.0;
  //   for (int i = 0; i < total; ++i) {
  //     double a = h_out_direct[i];
  //     double b = h_out_reuse[i];
  //     double abs_err = std::abs(a - b);
  //     max_abs = std::max(max_abs, abs_err);
  //   }

  //   printf("Direct exp(x_i*s):       %.6f ms/iter\n", ms_direct);
  //   printf("Reuse es + sqrt/square:  %.6f ms/iter\n", ms_reuse);
  //   printf("Speedup (direct/reuse):  %.4fx\n", ms_direct / ms_reuse);
  //   printf("Error vs direct: max_abs=%.6e,\n", max_abs);

  //   CUDA_CHECK(cudaFree(d_s));
  //   CUDA_CHECK(cudaFree(d_xi));
  //   CUDA_CHECK(cudaFree(d_out));

  float *d_dummy = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(float)));
  bench_pure_compute_cycles<<<1, 256>>>(d_dummy);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_dummy));
  return 0;
}