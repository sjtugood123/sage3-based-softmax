# example_sageattn3.py
import torch
import importlib
import math
from typing import Callable, Dict, List, Tuple

from sageattn3 import sageattn3_blackwell as sageattn3_baseline


def sageattn3_bypass(q, k, v, is_causal=False, per_block_mean=True):
    api_mod = importlib.import_module("sageattn3.api")
    fp4attn_cuda_bypass = importlib.import_module("fp4attn_cuda_bypass")
    old_mod = api_mod.fp4attn_cuda
    api_mod.fp4attn_cuda = fp4attn_cuda_bypass
    try:
        out = api_mod.sageattn3_blackwell(
            q, k, v, is_causal=is_causal, per_block_mean=per_block_mean
        )
    finally:
        api_mod.fp4attn_cuda = old_mod
    return out


def calc_metrics(a: torch.Tensor, b: torch.Tensor):
    diff = (a - b).float()
    abs_diff = diff.abs()
    rmse = torch.sqrt((diff * diff).mean()).item()
    rel_l2 = (torch.norm(diff).item() / (torch.norm(a.float()).item() + 1e-12))
    return {
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "rmse": rmse,
        "rel_l2": rel_l2,
    }


def benchmark(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(q, k, v)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times_ms = []

    with torch.no_grad():
        for _ in range(iters):
            starter.record()
            _ = fn(q, k, v)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))

    return sum(times_ms) / len(times_ms)


def benchmark_pipeline_breakdown(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    use_bypass: bool,
    warmup: int,
    iters: int,
):
    api_mod = importlib.import_module("sageattn3.api")
    if use_bypass:
        fp4attn_mod = importlib.import_module("fp4attn_cuda_bypass")
    else:
        fp4attn_mod = importlib.import_module("fp4attn_cuda")

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            q2, k2, v2, delta_s = api_mod.preprocess_qkv(q, k, v, per_block_mean=True)
            qlist = api_mod.scale_and_quant_fp4(q2)
            klist = api_mod.scale_and_quant_fp4_permute(k2)
            vlist = api_mod.scale_and_quant_fp4_transpose(v2)
            softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
            _ = fp4attn_mod.fwd(
                qlist[0], klist[0], vlist[0],
                qlist[1], klist[1], vlist[1],
                delta_s, k.size(2), None,
                softmax_scale, False, True, q.dtype == torch.bfloat16
            )
    torch.cuda.synchronize()

    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    t_pre_ms, t_quant_ms, t_core_ms, t_total_ms = [], [], [], []

    with torch.no_grad():
        for _ in range(iters):
            # preprocess
            e0.record()
            q2, k2, v2, delta_s = api_mod.preprocess_qkv(q, k, v, per_block_mean=True)
            e1.record(); torch.cuda.synchronize()
            t_pre_ms.append(e0.elapsed_time(e1))

            # quant
            e0.record()
            qlist = api_mod.scale_and_quant_fp4(q2)
            klist = api_mod.scale_and_quant_fp4_permute(k2)
            vlist = api_mod.scale_and_quant_fp4_transpose(v2)
            e1.record(); torch.cuda.synchronize()
            t_quant_ms.append(e0.elapsed_time(e1))

            # core attn
            softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
            e0.record()
            _ = fp4attn_mod.fwd(
                qlist[0], klist[0], vlist[0],
                qlist[1], klist[1], vlist[1],
                delta_s, k.size(2), None,
                softmax_scale, False, True, q.dtype == torch.bfloat16
            )
            e1.record()
            torch.cuda.synchronize()
            t_core_ms.append(e0.elapsed_time(e1))

            t_total_ms.append(t_pre_ms[-1] + t_quant_ms[-1] + t_core_ms[-1])

    def avg(xs): return sum(xs) / len(xs)
    return {
        "pre_ms": avg(t_pre_ms),
        "quant_ms": avg(t_quant_ms),
        "core_ms": avg(t_core_ms),
        "total_ms": avg(t_total_ms),
    }


def run_one_case(
    shape: Tuple[int, int, int, int, int],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    B, H, Lq, Lk, D = shape
    q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    k = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    v = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    warmup, iters = 10, 20

    # with torch.no_grad():
    #     out_base = sageattn3_baseline(q, k, v, is_causal=False, per_block_mean=True)
    #     out_bypass = sageattn3_bypass(q, k, v, is_causal=False, per_block_mean=True)

    # metrics = calc_metrics(out_base, out_bypass)

    t_base = benchmark(
        lambda a, b, c: sageattn3_baseline(a, b, c, is_causal=False, per_block_mean=True),
        q, k, v, warmup, iters
    )
    t_bypass = benchmark(
        lambda a, b, c: sageattn3_bypass(a, b, c, is_causal=False, per_block_mean=True),
        q, k, v, warmup, iters
    )

    speedup = t_base / t_bypass
    saved = (t_base - t_bypass) / t_base * 100.0

    base = benchmark_pipeline_breakdown(q, k, v, use_bypass=False, warmup=warmup, iters=iters)
    byp  = benchmark_pipeline_breakdown(q, k, v, use_bypass=True,  warmup=warmup, iters=iters)

    speedup_total = base["total_ms"] / byp["total_ms"]
    speedup_core = base["core_ms"] / byp["core_ms"]

    return {
        "shape": shape,
        "warmup": warmup,
        "iters": iters,
        "base_total_ms": base["total_ms"],
        "byp_total_ms": byp["total_ms"],
        "base_pre_ms": base["pre_ms"],
        "byp_pre_ms": byp["pre_ms"],
        "base_quant_ms": base["quant_ms"],
        "byp_quant_ms": byp["quant_ms"],
        "base_core_ms": base["core_ms"],
        "byp_core_ms": byp["core_ms"],
        "speedup_total": speedup_total,
        "speedup_core_no_quant": speedup_core,
    }


def has_nan_in_result(result: Dict) -> Tuple[bool, List[str]]:
    nan_keys: List[str] = []
    for k, v in result.items():
        if isinstance(v, float) and math.isnan(v):
            nan_keys.append(k)
    return (len(nan_keys) > 0, nan_keys)


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # 小 -> 中 -> 大（都满足 D=128）
    cases: List[Tuple[int, int, int, int, int]] = [
        # (1, 16, 1024, 1024, 128),
        # (1, 16, 2048, 2048, 128),
        #4k
        # (1, 32, 4096, 4096, 128),
        # (1, 32, 8192, 8192, 128),
        #16k
        # (1, 32, 16384, 16384, 128),
        (1, 32, 40960, 40960, 128),
        #128k:OOM
        # (1, 32, 131072, 131072, 128),
    ]

    print("==== SageAttention3 baseline vs bypass: timing ====\n")
    results = []
    for shape in cases:
        try:
            r = run_one_case(shape, device="cuda", dtype=torch.bfloat16)
            has_nan, nan_keys = has_nan_in_result(r)
            if has_nan:
                print(f"[NaN DETECTED] shape={shape}, fields={nan_keys}")
            results.append(r)
            # print(
            #     f"{r['shape']} | total: {r['base_total_ms']:.3f}->{r['byp_total_ms']:.3f} ms "
            #     f"({r['speedup_total']:.2f}x) | quant(base/byp): {r['base_quant_ms']:.3f}/{r['byp_quant_ms']:.3f} ms "
            #     f"| core(no-quant): {r['base_core_ms']:.3f}->{r['byp_core_ms']:.3f} ms "
            #     f"({r['speedup_core_no_quant']:.2f}x)"
            # )
        except RuntimeError as e:
            print(f"shape={shape} FAILED: {e}\n")

    print("==== Summary Table ====")
    for r in results:
        has_nan, nan_keys = has_nan_in_result(r)
        nan_tag = f" [NaN in {nan_keys}]" if has_nan else ""
        print(
            f"{r['shape']} | total: {r['base_total_ms']:.3f}->{r['byp_total_ms']:.3f} ms "
            f"({r['speedup_total']:.2f}x) | pre(base/byp): {r['base_pre_ms']:.3f}/{r['byp_pre_ms']:.3f} ms | quant(base/byp): {r['base_quant_ms']:.3f}/{r['byp_quant_ms']:.3f} ms "
            f"| core(no-quant,no-pre): {r['base_core_ms']:.3f}->{r['byp_core_ms']:.3f} ms "
            f"({r['speedup_core_no_quant']:.2f}x){nan_tag}"
        )


if __name__ == "__main__":
    main()