
#include "cute_extension.h"
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.h"
#include "utils.h"
#include <iostream>

using namespace cute;

int main() {
  // 1. 提取你定义的极简底层操作符
  using mma_op = SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4;

  // 2. 将其包装为 CuTe 的 MMA_Atom
  using mma_atom = MMA_Atom<mma_op>;

  // 3. 实例化一个 TiledMMA。
  // 这里我们不加任何 Layout 参数，它默认就是一个 Warp (32个线程) 的行为
  using TiledMma = TiledMMA<mma_atom>;
  TiledMma tiled_mma;

  std::cout << "========= 开始生成 TiledMMA 布局 LaTeX =========" << std::endl;
  // 这一行会直接生成 A, B, C 三个矩阵的宏观 LaTeX 代码
  print_latex(tiled_mma);

  std::cout << "\n========= 单独打印矩阵 A 的布局 =========" << std::endl;
  print_latex(tiled_mma.get_layout_A());

  std::cout << "\n========= 打印自定义的 SFA (缩放因子) 布局 ========="
            << std::endl;
  // 使用你代码里定义的专属函数，提取 Scale Factor 的 TV (Thread-Value) 布局
  auto layout_SFA = get_layoutSFA_TV(tiled_mma);
  print("SFA Layout: ");
  print(layout_SFA);
  std::cout << std::endl;

  return 0;
}