"""optimizer.py — Quản lý bộ tối ưu hoá và cắt gradient (gradient clipping).

Công thức cơ bản (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas: tuple[float, float] = (0.9, 0.999),
                    eps: float = 1e-8) -> torch.optim.Optimizer:
    """Trả về một torch.optim.Optimizer tương ứng."""
    name_clean = name.lower().strip()
    if name_clean not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' không hợp lệ. Chọn một trong: {OPTIMIZERS}")

    if name_clean == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name_clean == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name_clean == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name_clean == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Chưa hỗ trợ optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học (learning rate scheduler)."""
    if name is None or not name:
        return None

    name_clean = name.lower().strip()
    if name_clean == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    else:
        raise ValueError(f"Chưa hỗ trợ scheduler: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Lưu ý:
      - clip_grad_norm_ luôn trả về chuẩn L2 trước khi bị cắt (scale).
      - Nếu max_norm là None, ta truyền float('inf') để chỉ đo độ lớn gradient mà không cắt.
      - Khi dùng Mixed Precision (FP16), phải gọi scaler.unscale_(optimizer) trước hàm này.
    """
    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_norm))

    return float(total_norm)
