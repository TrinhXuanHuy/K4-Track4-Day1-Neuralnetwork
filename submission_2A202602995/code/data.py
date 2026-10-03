"""data.py — Xử lý dữ liệu cho Forest CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
EXPECTED_FEATURES = 54
EXPECTED_TRAIN = 464_809
EXPECTED_EVAL = 116_203


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    p_dir = Path(processed_dir)
    train_path = p_dir / "train.npz"
    eval_path = p_dir / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {train_path} hoặc {eval_path}. "
            "Hãy chạy `python scripts/split_data.py` trước!"
        )

    with np.load(train_path) as tr:
        X_train_full = tr["X"].astype(np.float32)
        y_train_full = tr["y"].astype(np.int64)

    with np.load(eval_path) as ev:
        X_eval = ev["X"].astype(np.float32)
        y_eval = ev["y"].astype(np.int64)
        eval_row_id = ev["row_id"].astype(np.int64)

    assert X_train_full.shape == (EXPECTED_TRAIN, EXPECTED_FEATURES), f"Sai shape train: {X_train_full.shape}"
    assert y_train_full.shape == (EXPECTED_TRAIN,), f"Sai shape y train: {y_train_full.shape}"
    assert X_eval.shape == (EXPECTED_EVAL, EXPECTED_FEATURES), f"Sai shape eval: {X_eval.shape}"
    assert y_eval.shape == (EXPECTED_EVAL,), f"Sai shape y eval: {y_eval.shape}"
    assert eval_row_id.shape == (EXPECTED_EVAL,), f"Sai shape eval_row_id: {eval_row_id.shape}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    """
    mean = X_tr[:, :N_NUMERIC].mean(axis=0).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0).astype(np.float32)
    # Tránh chia cho 0 nếu có cột không đổi
    std[std == 0.0] = 1.0
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_norm = X.copy()
    X_norm[:, :N_NUMERIC] = (X_norm[:, :N_NUMERIC] - mean) / std
    return X_norm


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval (y là int64)
    và các mảng numpy: eval_row_id, mean, std.
    """
    dev = torch.device(device) if isinstance(device, str) else device

    # 1. Nạp train/eval
    X_train_full, y_train_full, X_eval_raw, y_eval_raw, eval_row_id = load_split(processed_dir)

    # 2. Tách val từ train (phân tầng, seed cố định)
    X_tr_raw, y_tr_raw, X_val_raw, y_val_raw = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # 3. Chuẩn hoá: CHỈ fit trên train còn lại
    mean, std = fit_standardizer(X_tr_raw)
    X_tr_norm = apply_standardizer(X_tr_raw, mean, std)
    X_val_norm = apply_standardizer(X_val_raw, mean, std)
    X_eval_norm = apply_standardizer(X_eval_raw, mean, std)

    # 4. Chuyển sang Tensor và đưa lên device
    X_tr = torch.from_numpy(X_tr_norm).to(dtype=torch.float32, device=dev)
    y_tr = torch.from_numpy(y_tr_raw).to(dtype=torch.int64, device=dev)

    X_val = torch.from_numpy(X_val_norm).to(dtype=torch.float32, device=dev)
    y_val = torch.from_numpy(y_val_raw).to(dtype=torch.int64, device=dev)

    X_eval = torch.from_numpy(X_eval_norm).to(dtype=torch.float32, device=dev)
    y_eval = torch.from_numpy(y_eval_raw).to(dtype=torch.int64, device=dev)

    # 5. In thống kê kiểm tra
    val_majority_class = int(torch.bincount(y_val).argmax().item())
    majority_acc = float((y_val == val_majority_class).float().mean().item())

    print(f"Data prepared on {dev}:")
    print(f"  Train : X={X_tr.shape}, y={y_tr.shape}")
    print(f"  Val   : X={X_val.shape}, y={y_val.shape}")
    print(f"  Eval  : X={X_eval.shape}, y={y_eval.shape}")
    print(f"  Majority class baseline on Val: class {val_majority_class} -> acc = {majority_acc:.4f}")

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "device": dev,
    }


def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                    generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay thế DataLoader.
    
    Tự xáo trộn tensor trên device để huấn luyện nhanh và tốn ít bộ nhớ.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
