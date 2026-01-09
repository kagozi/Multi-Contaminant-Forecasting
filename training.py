# ============================================================================
# Module 4: training.py - Training and evaluation
# ============================================================================

import numpy as np
import torch
from tqdm import tqdm


def masked_mse(y_hat: torch.Tensor, y: torch.Tensor, m: torch.Tensor, eps: float = 1e-8):
    """Masked MSE loss"""
    diff2 = (y_hat - y) ** 2
    return (diff2 * m).sum() / m.sum().clamp_min(eps)


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.mean(np.abs(y_true - y_pred)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Normalized RMSE"""
    rmse_val = rmse(y_true, y_pred)
    r = float(y_true.max()) - float(y_true.min())
    return float(rmse_val / (r + 1e-8))


@torch.no_grad()
def eval_loop(model, loader, device):
    """Evaluate on a dataloader"""
    model.eval()
    losses = []
    for batch in loader:
        y_hat = model(
            batch["basin_id"].to(device),
            batch["x_static"].to(device),
            batch["coord"].to(device),
            batch["seq_in"].to(device),
        )
        loss = masked_mse(y_hat, batch["y_target"].to(device), batch["m_target"].to(device))
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(model, dl_train, dl_val, dl_test, cfg, device):
    """Train for cfg.epochs and return history"""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(dl_train, desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
            y_hat = model(
                batch["basin_id"].to(device),
                batch["x_static"].to(device),
                batch["coord"].to(device),
                batch["seq_in"].to(device),
            )
            loss = masked_mse(y_hat, batch["y_target"].to(device), batch["m_target"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))

        val_loss = eval_loop(model, dl_val, device)
        test_loss = eval_loop(model, dl_test, device)

        row = {"epoch": epoch, "train_loss": np.mean(losses), "val_loss": val_loss, "test_loss": test_loss}
        history.append(row)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_loss={row['train_loss']:.6f}, val_loss={val_loss:.6f}, test_loss={test_loss:.6f}")

    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_on_loader(model, loader, device):
    """Get predictions on a loader"""
    model.eval()
    y_trues, y_preds, masks = [], [], []
    for batch in loader:
        y_hat = model(
            batch["basin_id"].to(device),
            batch["x_static"].to(device),
            batch["coord"].to(device),
            batch["seq_in"].to(device),
        )
        y_trues.append(batch["y_target"].cpu().numpy())
        y_preds.append(y_hat.cpu().numpy())
        masks.append(batch["m_target"].cpu().numpy())
    return np.concatenate(y_trues), np.concatenate(y_preds), np.concatenate(masks)