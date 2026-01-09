"""
Spatio-Temporal Multi-Contaminant Forecasting - Complete Training Pipeline

Usage:
    python main.py
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TrainConfig, seed_everything
from data import (
    make_synthetic_watershed_multicontaminant,
    make_sparsity_masks,
    SpatioTemporalMultiContaminantDataset,
)
from model import SpatialGRUMultiContaminantForecaster
from training import train_model, predict_on_loader, rmse, mae, nrmse
from plots import save_all_plots


def main():
    """Main training pipeline"""
    
    # ===== SETUP =====
    print("\n" + "="*70)
    print("SETUP")
    print("="*70)
    
    cfg = TrainConfig()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    
    print(f"Device: {device}")
    print(f"Contaminants: {cfg.contaminants}")
    print(f"Basins: {cfg.synth_basins}, Days: {cfg.synth_days}")
    print(f"Spatial coverage: {cfg.p_space*100:.0f}%, Temporal coverage: {cfg.p_time*100:.0f}%")
    
    # ===== DATA GENERATION =====
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC DATA")
    print("="*70)
    
    syn = make_synthetic_watershed_multicontaminant(
        cfg.synth_basins,
        cfg.synth_days,
        cfg.synth_dyn_features,
        cfg.synth_static_features,
        seed=cfg.seed,
    )
    
    X_dyn = syn["X_dyn"]
    X_static = syn["X_static"]
    coords = syn["coords"]
    y_dict = {
        "no3": syn["y_no3"],
        "phosphorus": syn["y_phosphorus"],
        "discharge": syn["y_discharge"],
    }
    m_obs = syn["m_obs"]
    
    B, T, F = X_dyn.shape
    S = X_static.shape[1]
    
    print(f"✓ Data shape: B={B}, T={T}, F={F}, S={S}")
    for name, y in y_dict.items():
        print(f"  {name}: {y.min():.3f} - {y.max():.3f} mg/L")
    
    # ===== SPARSITY MASKS =====
    print("\n" + "="*70)
    print("CREATING SPARSITY MASKS")
    print("="*70)
    
    sensor_mask, m_train = make_sparsity_masks(m_obs, cfg.p_space, cfg.p_time, cfg.seed)
    print(f"✓ Instrumented basins: {int(sensor_mask.sum())}/{B}")
    print(f"✓ Temporal coverage: {cfg.p_time*100:.0f}%")
    
    # ===== DATASETS & LOADERS =====
    print("\n" + "="*70)
    print("CREATING DATASETS & DATALOADERS")
    print("="*70)
    
    ds_train = SpatioTemporalMultiContaminantDataset(
        X_dyn, X_static, coords, y_dict, m_train,
        cfg.contaminants, cfg.history_len, cfg.horizon, "train",
        cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed,
    )
    ds_val = SpatioTemporalMultiContaminantDataset(
        X_dyn, X_static, coords, y_dict, m_train,
        cfg.contaminants, cfg.history_len, cfg.horizon, "val",
        cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed,
    )
    ds_test = SpatioTemporalMultiContaminantDataset(
        X_dyn, X_static, coords, y_dict, m_train,
        cfg.contaminants, cfg.history_len, cfg.horizon, "test",
        cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed,
    )
    
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    print(f"✓ Train samples: {len(ds_train)}")
    print(f"✓ Val samples: {len(ds_val)}")
    print(f"✓ Test samples: {len(ds_test)}")
    
    # ===== MODEL =====
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    f_in = F + len(cfg.contaminants) + 1
    model = SpatialGRUMultiContaminantForecaster(
        num_basins=B,
        f_in=f_in,
        static_dim=S,
        num_contaminants=len(cfg.contaminants),
        d_basin=cfg.d_basin,
        d_static=cfg.d_static,
        d_coord=cfg.d_coord,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {num_params:,}")
    
    # ===== TRAINING =====
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history = train_model(model, dl_train, dl_val, dl_test, cfg, device)
    
    # ===== EVALUATION =====
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    y_true, y_pred, m = predict_on_loader(model, dl_test, device)
    idx = m[:, 0] > 0.5
    
    metrics = {"test_supervised_points": int(idx.sum())}
    
    print(f"\nTest Set Metrics (Per-Contaminant):")
    print("-" * 70)
    for i, cont in enumerate(cfg.contaminants):
        y_t = y_true[idx, i].astype(np.float64)
        y_p = y_pred[idx, i].astype(np.float64)
        
        rmse_c = rmse(y_t, y_p)
        mae_c = mae(y_t, y_p)
        nrmse_c = nrmse(y_t, y_p)
        
        metrics[f"{cont}_rmse"] = float(rmse_c)
        metrics[f"{cont}_mae"] = float(mae_c)
        metrics[f"{cont}_nrmse"] = float(nrmse_c)
        
        print(f"{cont.upper():15s} | RMSE: {rmse_c:.4f} | MAE: {mae_c:.4f} | NRMSE: {nrmse_c:.4f}")
    
    print("-" * 70)
    print(json.dumps(metrics, indent=2))
    
    # ===== SAVE RESULTS =====
    os.makedirs("results", exist_ok=True)
    
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to results/metrics.json")
    
    with open("results/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"✓ History saved to results/history.json")
    
    # ===== PLOTS =====
    df_metrics = save_all_plots(
        model, history, y_true, y_pred, m, ds_test,
        y_dict, coords, cfg, device, output_dir="figures"
    )
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Model trained on {B} basins over {T} days")
    print(f"✓ Forecasting {len(cfg.contaminants)} contaminants: {cfg.contaminants}")
    print(f"✓ 7-day prediction horizon with {cfg.history_len}-day input window")
    print(f"✓ Results saved to: results/ and figures/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()