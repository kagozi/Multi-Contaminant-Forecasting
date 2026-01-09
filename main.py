from config import TrainConfig, seed_everything
from data import make_synthetic_watershed_multicontaminant, make_sparsity_masks, SpatioTemporalMultiContaminantDataset
from model import SpatialGRUMultiContaminantForecaster
from training import train_model, predict_on_loader, rmse, mae, nrmse
import torch
from torch.utils.data import DataLoader

# Setup
cfg = TrainConfig()
seed_everything(cfg.seed)
device = torch.device(cfg.device)

# Generate synthetic data
syn = make_synthetic_watershed_multicontaminant(cfg.synth_basins, cfg.synth_days, 
                                                cfg.synth_dyn_features, cfg.synth_static_features)
X_dyn, X_static, coords = syn["X_dyn"], syn["X_static"], syn["coords"]
y_dict = {"no3": syn["y_no3"], "phosphorus": syn["y_phosphorus"], "discharge": syn["y_discharge"]}
m_obs = syn["m_obs"]

# Create sparsity
_, m_train = make_sparsity_masks(m_obs, cfg.p_space, cfg.p_time, cfg.seed)

# Create datasets & loaders
ds_train = SpatioTemporalMultiContaminantDataset(X_dyn, X_static, coords, y_dict, m_train, 
    cfg.contaminants, cfg.history_len, cfg.horizon, "train", cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
ds_val = SpatioTemporalMultiContaminantDataset(X_dyn, X_static, coords, y_dict, m_train, 
    cfg.contaminants, cfg.history_len, cfg.horizon, "val", cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
ds_test = SpatioTemporalMultiContaminantDataset(X_dyn, X_static, coords, y_dict, m_train, 
    cfg.contaminants, cfg.history_len, cfg.horizon, "test", cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)

dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

# Create model
B, F = X_dyn.shape[0], X_dyn.shape[2]
f_in = F + len(cfg.contaminants) + 1
model = SpatialGRUMultiContaminantForecaster(
    num_basins=B, f_in=f_in, static_dim=X_static.shape[1],
    num_contaminants=len(cfg.contaminants),
    d_basin=cfg.d_basin, d_static=cfg.d_static, d_coord=cfg.d_coord,
    d_model=cfg.d_model, dropout=cfg.dropout
).to(device)

# Train
history = train_model(model, dl_train, dl_val, dl_test, cfg, device)

# Evaluate
y_true, y_pred, m = predict_on_loader(model, dl_test, device)
idx = m[:, 0] > 0.5

for i, cont in enumerate(cfg.contaminants):
    y_t = y_true[idx, i]
    y_p = y_pred[idx, i]
    print(f"{cont}: RMSE={rmse(y_t, y_p):.4f}, MAE={mae(y_t, y_p):.4f}, NRMSE={nrmse(y_t, y_p):.4f}")