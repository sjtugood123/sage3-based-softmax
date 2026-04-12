/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cmath>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "utils.h"

namespace flash {

using namespace cute;

// template <class T>
// struct PrintType;

template <int Rows>//rows=2
struct new_SoftmaxFused{

    using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
    TensorT row_sum, row_max, scores_scale;
    static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
    static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f; //log2f(1/(6*448))
    static constexpr float fp4_scale_log2 = -2.584962500721156f; // log2f(1/fp4_scale)
    static constexpr int RowReductionThr = 4;

    CUTLASS_DEVICE SoftmaxFused(){};

    template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE auto online_softmax_with_quant(
        TensorAcc& acc, 
        TensorMax& AbsMaxP,
        // softmax_scale_log2=log2e÷√d
        // 因为e^x=2^(x*log2e)，log2e是为了把exp换成exp2
        // √d是softmax的缩放
        const float softmax_scale_log2
    ) {
        
/*
acc_conversion_view:
(( (2, 4), (2, 2) ), 1, 2)
acc_conversion_flatten:
((2, 4), (2, 2, 1, 2))
*/

        // 这三个是同一块寄存器，只是视图不同
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));
        Tensor acc_conversion_view = make_tensor(acc.data(), flash::convert_to_conversion_layout(acc.layout()));
        Tensor acc_conversion_flatten = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));

        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1); // exchange max with neighbour thread of 8 elements
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
                    AbsMaxP(mi, ni) /= 6.0f;

                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                // const float max_scaled = InfCheck//IngCheck默认False
                //                         ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                //                         : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
                const float new_max_scaled = row_max(mi) * softmax_scale_log2;
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    // 这样计算会有重复，因为相邻两个线程的AbsMaxP(mi, ni)是一样的，先这样写，后面可能再试试shfl会不会有优化
                    // 这样计算的正确性是有问题的，因为得用e^(x*s-rowmax)，但是现在算的是e^(s-rowmax)
                    AbsMaxP(mi, ni) = flash::ptx_exp2(AbsMaxP(mi, ni) * softmax_scale_log2 - new_max_scaled);//es√d
                    float inv_es = 1.0f / AbsMaxP(mi, sfi);
                    float inv_sqrt_es = rsqrtf(AbsMaxP(mi, sfi));

                    // 固定计算 + 表查
                    float inv_es2 = inv_es * inv_es;          // e^-2s
                    float inv_es3 = inv_es2 * inv_es;         // e^-3s
                    float inv_es4 = inv_es2 * inv_es2;        // e^-4s
                    float inv_es6 = inv_es4 * inv_es2;        // e^-6s
                    float inv_es_sqrt = inv_es * inv_sqrt_es; // e^-1.5s

                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        //根据x/s的值选择不同的指数
                        acc_reduction_view(mi, make_coord(ei, ni)) = acc_reduction_view(mi, make_coord(ei, ni)) / AbsMaxP(mi, ni);
                        // todo: 这里先不quant到fp4，但是quant之后应该可以通过位运算比较快的定位到应该取哪一个值
                        // 不quant好像不太行，要quant一下，舍入啥的不是很明确
                    }

                    // AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
                
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    //计算当前块的 $e^{x-m}$
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                }
                
            }
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
        else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1); // exchange max with neighbour thread of 8 elements
                    AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                float scores_max_cur = !InfCheck
                                        ? row_max(mi)
                                        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = flash::ptx_exp2((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
                row_sum(mi) = row_sum(mi) * scores_scale(mi);
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
                }
                // scores_scale(mi) = max_scaled;
            }
        }
        //?
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            // 其实AbsMaxP 是基于 acc 的 layout make出来的，和acc_conversion_flatten的layout具有高度相似性
            // 所以AbsMaxP 的第 i 个元素，在物理地址上，天然就对应着 acc 中第 i 个逻辑块的缩放因子。
            //acc_conversion_flatten：((AtomN, AtomM), (MmaN_divided, MmaM, _2))?
            for (int j = 0; j < size<0>(acc_conversion_flatten); ++j) {
            // 在softmax过程中进行量化的步骤是否会损失精度需要仔细想想，量化的那个过程也要看看熟悉一下
                acc_conversion_flatten(j, i) /= AbsMaxP(i);
            }
        }
    }

    //finalize  是在所有块都处理完、总和已经完全确定之后，最后执行的那一次“大归一化”。
    //rescale_o 是在处理不同数据块（Tile）时，因为最大值变了而进行的“局部修正”。
    template<typename TensorAcc>
    CUTLASS_DEVICE void finalize(TensorAcc& o_store) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        CUTLASS_PRAGMA_UNROLL
        // size(row_max)是当前线程负责的行数
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            // cute里面访问tensor的某一个元素是用括号，如row_sum(mi)
            // RowReductionThr 是 4
            // 线程t和线程t+i交换
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                // -1即0xFFFFFFFF，表示所有线程都要参与交换
                // 这里只交换了i=1和i=2，为什么呢?四个线程就处理了一行的数据?
                // 但为什么是4?编程的时候设计的还是规定好的
                float sum_recv = __shfl_xor_sync(int32_t(-1), row_sum(mi), i);
                row_sum(mi) += sum_recv;
            }
            float sum = row_sum(mi);
            // NaN会造成sum != sum，所以这里是在检查NaN
            // 这里多一次除法，后面对这一行每一个元素都做一次乘法，比每个元素都做一次除法更快
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
            CUTLASS_PRAGMA_UNROLL
            // ni:column, size<1>(o_store_reduction_view)是每行的元素个数
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) *= inv_sum;
            }
        }
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        Tensor o_tmp_reduction_view = make_tensor(o_tmp.data(), flash::convert_to_reduction_layout(o_tmp.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) = o_store_reduction_view(mi, ni) * scores_scale(mi) + o_tmp_reduction_view(mi, ni);
             }
        }

    }


};
} // namespace flash