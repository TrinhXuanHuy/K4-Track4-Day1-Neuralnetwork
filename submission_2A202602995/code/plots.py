"""plots.py — Vẽ biểu đồ huấn luyện và so sánh thí nghiệm.

Sản phẩm bắt buộc theo README mục 6:
  - Mỗi thí nghiệm một ảnh `figures/<exp_id>.png` (3 ô: loss, acc/f1, grad_norm).
  - Ảnh so sánh nhóm `figures/compare_<nhóm>.png`.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
      (1) train_loss và val_loss theo epoch
      (2) val_acc và val_macro_f1 theo epoch
      (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    cfg = result["cfg"]
    hist = result["history"]
    sumry = result["summary"]
    epochs = hist["epoch"]

    if not epochs:
        print(f"Bỏ qua vẽ {path} do không có dữ liệu epoch.")
        return

    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    exp_id = cfg.get("exp_id", "exp")
    opt = cfg.get("optimizer", "")
    lr = cfg.get("lr", "")
    best_ep = sumry.get("best_epoch", -1)

    # Ô 1: Train & Val Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", lw=2)
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    if best_ep != -1:
        axes[0].axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best ep ({best_ep})")
    axes[0].set_title(f"Loss ({exp_id})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    # Ô 2: Val Accuracy & Macro-F1
    axes[1].plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", lw=2)
    axes[1].plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#9467bd", lw=2)
    if best_ep != -1:
        axes[1].axvline(best_ep, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_title(f"Accuracy & Macro-F1\n(Best F1={sumry.get('val_macro_f1', 0):.4f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    # Ô 3: Gradient Norm (trước khi clip)
    axes[2].plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="#d62728", lw=1.5)
    axes[2].set_title(f"Gradient Norm\n(opt={opt}, lr={lr})", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L2 Norm")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=9)

    plt.suptitle(f"Experiment: {exp_id} | {cfg.get('description', '')}", fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục để so sánh trực quan."""
    if not results:
        return

    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for r in results:
        cfg = r["cfg"]
        hist = r["history"]
        exp_id = cfg.get("exp_id", "exp")
        if metric in hist and hist[metric]:
            ax.plot(hist["epoch"], hist[metric], label=exp_id, lw=2)

    metric_name_map = {
        "val_loss": "Validation Loss",
        "train_loss": "Train Loss",
        "val_acc": "Validation Accuracy",
        "val_macro_f1": "Validation Macro-F1",
        "grad_norm": "Gradient Norm (pre-clip)",
    }

    display_name = metric_name_map.get(metric, metric)
    ax.set_title(title or f"So sánh {display_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(display_name, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu biểu đồ so sánh: {out_file}")
