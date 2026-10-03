"""train.py — Quản lý vòng lặp huấn luyện, đánh giá và thí nghiệm.

Mọi thí nghiệm chỉ cần thay đổi dict cấu hình (cfg) và gọi `run_experiment(cfg, data)`.
Tất cả các độ đo (loss, accuracy, macro-F1) đều đồng nhất với `scripts/evaluate.py`.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base (54->256->128->7)",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn bằng tập val
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,            # None = không clip; hoặc số thực (ví dụ 1.0)
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cố định cho random, numpy, torch CPU và GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng (không trọng số) của F1 7 lớp.
    
    F1_c = 2 * P_c * R_c / (P_c + R_c), bằng 0 nếu P_c + R_c == 0.
    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = nhãn dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax(logits) ở chế độ eval mode."""
    model.eval()
    preds_list = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds_list.append(logits.argmax(dim=-1))
    return torch.cat(preds_list, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Tính hàm mất mát:
      - 'ce' : CrossEntropyLoss trên logits thô và nhãn int64
      - 'mse': Mean Squared Error giữa logits và one-hot encoding của y
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Chưa hỗ trợ hàm mất mát: {loss_name}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Đánh giá mô hình ở chế độ eval() (dropout tắt) và no_grad.

    Trả về: {"loss": float, "acc": float, "macro_f1": float}
    """
    model.eval()
    n = len(X)
    total_loss = 0.0
    all_preds = []

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += float(loss.item()) * len(xb)
        all_preds.append(logits.argmax(dim=-1))

    preds = torch.cat(all_preds, dim=0)
    acc = float((preds == y).float().mean().item())

    # Dựng ma trận nhầm lẫn 7x7 (chuyển sang CPU NumPy)
    y_true_np = y.cpu().numpy()
    y_pred_np = preds.cpu().numpy()
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_true_np, y_pred_np), 1)

    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": total_loss / n,
        "acc": acc,
        "macro_f1": macro_f1,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện mô hình theo dict cấu hình và trả về toàn bộ lịch sử + summary.

    Tất cả các đại lượng cần đo theo GUIDE và RUBRIC:
      - step0_loss (trước bước cập nhật đầu tiên)
      - train_loss (đo ở chế độ eval)
      - val_loss, val_acc, val_macro_f1
      - grad_norm trung bình (đo TRƯỚC khi clip)
      - thời gian mỗi epoch và bộ nhớ GPU cực đại
    """
    set_seed(cfg["seed"])
    dev = data["device"]
    is_cuda = (dev.type == "cuda")

    # 1. Khởi tạo mô hình
    hidden = tuple(cfg["hidden"])
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")
    model = MLP(hidden=hidden, dropout=dropout, init=init).to(dev)

    expected_p = EXPECTED_PARAMS.get(hidden)
    if expected_p is not None:
        assert count_params(model) == expected_p, (
            f"Số tham số không khớp: có {count_params(model)}, kỳ vọng {expected_p}"
        )

    # 2. Khởi tạo optimizer
    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9),
    )

    # 3. Thiết lập Precision (FP16 / BF16 / FP32)
    precision = cfg.get("precision", "fp32").lower()
    scaler = None
    autocast_dtype = torch.float32

    if precision == "fp16":
        if is_cuda:
            scaler = torch.amp.GradScaler("cuda")
            autocast_dtype = torch.float16
        else:
            print("Cảnh báo: FP16 yêu cầu CUDA GPU. Chuyển về FP32.")
            precision = "fp32"
    elif precision == "bf16":
        if is_cuda and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            print("Cảnh báo: BF16 không được hỗ trợ trên thiết bị này. Chuyển về FP32.")
            precision = "fp32"

    use_autocast = (precision in ("fp16", "bf16") and is_cuda)

    # 4. Đo Loss bước 0 trên tập Val (eval mode, trước khi update bất kỳ trọng số nào)
    step0_val = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_val["loss"]

    # Tập con train cố định để đo train loss nhanh & nhất quán ở eval mode
    n_train_sub = min(len(data["X_tr"]), 50_000)
    X_tr_sub = data["X_tr"][:n_train_sub]
    y_tr_sub = data["y_tr"][:n_train_sub]

    # Khởi tạo các cấu trúc lưu lịch sử
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    diverged = False

    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch"])
    clip_norm = cfg.get("clip_norm", None)

    # Tạo generator cho việc xáo lô
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"])

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    for epoch in range(1, epochs + 1):
        if is_cuda:
            torch.cuda.synchronize(dev)
        t_start = time.perf_counter()

        model.train()
        epoch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            # Kiểm tra NaN/Inf (phát hiện phân kỳ sớm)
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                print(f"Epoch {epoch}: Mô hình phân kỳ (loss = {loss.item()}). Dừng sớm.")
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                # scaler.unscale_ TRƯỚC KHI đo và clip gradient
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                epoch_grad_norms.append(gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                epoch_grad_norms.append(gn)
                optimizer.step()

        if is_cuda:
            torch.cuda.synchronize(dev)
        t_epoch = time.perf_counter() - t_start

        if diverged:
            break

        # Đánh giá cuối epoch ở chế độ eval()
        train_eval = evaluate(model, X_tr_sub, y_tr_sub, loss_name=cfg["loss"])
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        mean_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(t_epoch)

        # Lưu best checkpoint theo val_loss thấp nhất
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Tính peak memory (nếu có GPU)
    peak_mem_MB = 0.0
    if is_cuda:
        peak_mem_MB = float(torch.cuda.max_memory_allocated(dev) / (1024 * 1024))

    # Tóm tắt kết quả tại best_epoch
    if best_epoch != -1:
        best_idx = best_epoch - 1
        summary = {
            "step0_loss": float(step0_loss),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "val_acc": float(history["val_acc"][best_idx]),
            "val_macro_f1": float(history["val_macro_f1"][best_idx]),
            "time_per_epoch_s": float(np.mean(history["epoch_time_s"])),
            "peak_mem_MB": peak_mem_MB,
            "diverged": diverged,
        }
    else:
        summary = {
            "step0_loss": float(step0_loss),
            "best_val_loss": float("nan"),
            "best_epoch": -1,
            "final_train_loss": float("nan"),
            "final_val_loss": float("nan"),
            "val_acc": 0.0,
            "val_macro_f1": 0.0,
            "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
            "peak_mem_MB": peak_mem_MB,
            "diverged": True,
        }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file dự đoán định dạng `row_id,pred` cho `scripts/evaluate.py`."""
    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    df.to_csv(out_file, index=False)
    print(f"Đã lưu file dự đoán eval: {out_file} ({len(df):,} dòng)")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Chạy đánh giá cho mô hình cuối cùng (hoặc baseline) trên tập eval."""
    model = MLP(
        hidden=tuple(cfg["hidden"]),
        dropout=0.0,  # Luôn tắt dropout khi đánh giá cuối
        init=cfg.get("init", "he"),
    ).to(data["device"])

    if result["best_state"] is not None:
        model.load_state_dict(result["best_state"])
    else:
        print("Cảnh báo: không có best_state, dùng trọng số hiện tại của model.")

    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()
    write_predictions(data["eval_row_id"], preds_np, pred_path)
