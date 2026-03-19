# example_sageattn3.py
import torch
import importlib

from sageattn3 import sageattn3_blackwell as sageattn3_baseline

# 复用现有 API 实现逻辑，仅替换底层扩展模块为 bypass
# 前提：你已经编出了 fp4attn_cuda_bypass
def sageattn3_bypass(q, k, v, is_causal=False, per_block_mean=True):
    api_mod = importlib.import_module("sageattn3.api")
    fp4attn_cuda_bypass = importlib.import_module("fp4attn_cuda_bypass")

    # monkey patch：把 api.py 里的 fp4attn_cuda 临时替换成 bypass 模块
    old_mod = api_mod.fp4attn_cuda
    api_mod.fp4attn_cuda = fp4attn_cuda_bypass
    try:
        out = api_mod.sageattn3_blackwell(
            q, k, v,
            is_causal=is_causal,
            per_block_mean=per_block_mean
        )
    finally:
        api_mod.fp4attn_cuda = old_mod
    return out


def calc_metrics(a: torch.Tensor, b: torch.Tensor):
    diff = (a - b).float()
    abs_diff = diff.abs()

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    rmse = torch.sqrt((diff * diff).mean()).item()

    denom = torch.norm(a.float()).item()
    rel_l2 = (torch.norm(diff).item() / (denom + 1e-12))

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "relative_l2": rel_l2,
    }


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = "cuda"
    dtype = torch.bfloat16  # 或 torch.float16

    # 你可以多测几组
    B, H, Lq, Lk, D = 2, 16, 256, 256, 128

    q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    k = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    v = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    with torch.no_grad():
        out_base = sageattn3_baseline(q, k, v, is_causal=False, per_block_mean=True)
        out_bypass = sageattn3_bypass(q, k, v, is_causal=False, per_block_mean=True)

    metrics = calc_metrics(out_base, out_bypass)

    print("baseline:", out_base.shape, out_base.dtype)
    print("bypass  :", out_bypass.shape, out_bypass.dtype)
    print("---- error metrics (baseline vs bypass) ----")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.6e}")


if __name__ == "__main__":
    main()