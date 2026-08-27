from __future__ import annotations

import torch


def fit_low_rank_delta(
    x: torch.Tensor,
    delta_y: torch.Tensor,
    rank: int,
    ridge: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual ridge solution followed by truncated SVD in PEFT A/B convention."""
    if x.ndim != 2 or delta_y.ndim != 2 or x.shape[0] != delta_y.shape[0]:
        raise ValueError("x and delta_y need matching [samples, features] shapes")
    if rank <= 0:
        raise ValueError("rank must be positive")
    compute_dtype = torch.float64 if x.device.type == "cpu" else torch.float32
    x_work = x.to(compute_dtype)
    y_work = delta_y.to(compute_dtype)
    gram = x_work @ x_work.T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    dual = torch.linalg.solve(gram + ridge * identity, y_work)
    delta_w_in_out = x_work.T @ dual
    u, singular_values, vh = torch.linalg.svd(delta_w_in_out, full_matrices=False)
    actual_rank = min(rank, singular_values.numel())
    sqrt_s = singular_values[:actual_rank].clamp_min(0).sqrt()
    a = sqrt_s[:, None] * u[:, :actual_rank].T
    b = vh[:actual_rank].T * sqrt_s[None, :]
    return a.to(x.dtype), b.to(x.dtype)

