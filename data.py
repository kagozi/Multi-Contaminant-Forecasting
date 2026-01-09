import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def make_synthetic_watershed_multicontaminant(B: int, T: int, F: int, S: int, seed: int = 42):
    """
    Generate synthetic multi-contaminant watershed data.
    
    Returns:
        dict with X_dyn, X_static, coords, y_no3, y_phosphorus, y_discharge, m_obs, dates
    """
    rng = np.random.default_rng(seed)

    # Spatial setup
    lats = 35 + 10 * rng.random(B)
    lons = -120 + 15 * rng.random(B)
    coords = np.stack([lats, lons], axis=1).astype(np.float32)
    X_static = rng.normal(0, 1, size=(B, S)).astype(np.float32)

    # Temporal drivers
    t = np.arange(T)
    doy = (t % 365) / 365.0
    sin_doy = np.sin(2 * np.pi * doy)
    cos_doy = np.cos(2 * np.pi * doy)

    precip = rng.gamma(shape=2.0, scale=1.0, size=(B, T)).astype(np.float32)
    temp = (10 + 15 * sin_doy)[None, :] + rng.normal(0, 2.0, size=(B, T)).astype(np.float32)
    basin_scale = (0.5 + 0.5 * (1 / (1 + np.exp(-X_static[:, 0]))))[:, None].astype(np.float32)
    q = np.maximum((0.2 * precip + 0.05 * np.maximum(temp, 0) + 0.1 * rng.normal(0, 1, size=(B, T))).astype(np.float32) * basin_scale, 0.0)

    driver4 = (sin_doy[None, :] + 0.1 * rng.normal(0, 1, size=(B, T))).astype(np.float32)
    driver5 = (cos_doy[None, :] + 0.1 * rng.normal(0, 1, size=(B, T))).astype(np.float32)

    feats = np.stack([precip, temp, q, driver4, driver5, rng.normal(0, 1, size=(B, T)).astype(np.float32)], axis=-1)
    X_dyn = feats[:, :, :F].astype(np.float32)

    # Spatial hotspots
    lat0_no3, lon0_no3 = 40.0, -112.0
    dist2_no3 = (coords[:, 0] - lat0_no3) ** 2 + (coords[:, 1] - lon0_no3) ** 2
    hotspot_no3 = np.exp(-dist2_no3 / 30.0).astype(np.float32)

    lat0_p, lon0_p = 38.5, -114.0
    dist2_p = (coords[:, 0] - lat0_p) ** 2 + (coords[:, 1] - lon0_p) ** 2
    hotspot_p = np.exp(-dist2_p / 25.0).astype(np.float32)

    land = (1 / (1 + np.exp(-X_static[:, 1]))).astype(np.float32)

    # NO3 dynamics
    y_no3 = np.zeros((B, T), dtype=np.float32)
    y_no3[:, 0] = 2.0 + 3.0 * hotspot_no3 + 1.5 * land + 0.1 * rng.normal(size=B).astype(np.float32)
    for k in range(1, T):
        runoff = X_dyn[:, k, 0] + 0.5 * X_dyn[:, k, 2]
        temp_k = X_dyn[:, k, 1]
        decay_no3 = 0.02 + 0.01 * (temp_k > 15).astype(np.float32)
        y_no3[:, k] = ((1 - decay_no3) * y_no3[:, k - 1] + 0.08 * runoff + 0.3 * hotspot_no3 + 0.15 * land + 0.05 * rng.normal(size=B).astype(np.float32))
        y_no3[:, k] = np.maximum(y_no3[:, k], 0.0)

    # Phosphorus dynamics
    y_p = np.zeros((B, T), dtype=np.float32)
    y_p[:, 0] = 0.05 + 0.08 * hotspot_p + 0.05 * land + 0.02 * rng.normal(size=B).astype(np.float32)
    for k in range(1, T):
        runoff = X_dyn[:, k, 0] + 0.5 * X_dyn[:, k, 2]
        temp_k = X_dyn[:, k, 1]
        decay_p = 0.005 + 0.003 * (temp_k > 20).astype(np.float32)
        seasonal_p = 0.5 + 0.5 * sin_doy[k]
        y_p[:, k] = ((1 - decay_p) * y_p[:, k - 1] + 0.04 * runoff * seasonal_p + 0.2 * hotspot_p + 0.08 * land + 0.02 * rng.normal(size=B).astype(np.float32))
        y_p[:, k] = np.maximum(y_p[:, k], 0.0)

    # Discharge dynamics
    y_q = np.zeros((B, T), dtype=np.float32)
    for k in range(T):
        if k == 0:
            y_q[:, k] = (0.1 + 0.5 * X_dyn[:, k, 0] + 0.1 * X_dyn[:, k, 1] + 0.1 * rng.normal(size=B)).astype(np.float32)
        else:
            y_q[:, k] = (0.7 * y_q[:, k-1] + 0.3 * X_dyn[:, k, 0] + 0.1 * rng.normal(size=B)).astype(np.float32)
        y_q[:, k] = np.maximum(y_q[:, k], 0.0)

    # Irregular observations
    m_obs = (rng.random((B, T, 1)) < 0.1).astype(np.float32)

    return {
        "X_dyn": X_dyn,
        "X_static": X_static,
        "coords": coords,
        "y_no3": y_no3[:, :, None].astype(np.float32),
        "y_phosphorus": y_p[:, :, None].astype(np.float32),
        "y_discharge": y_q[:, :, None].astype(np.float32),
        "m_obs": m_obs,
        "dates": t.astype(np.int32),
    }


def make_sparsity_masks(m_obs: np.ndarray, p_space: float, p_time: float, seed: int):
    """Create spatial and temporal sparsity masks"""
    rng = np.random.default_rng(seed)
    B, T, _ = m_obs.shape

    B_sensor = max(1, int(p_space * B))
    sensor_idx = rng.choice(B, size=B_sensor, replace=False)
    sensor_mask = np.zeros((B,), dtype=np.float32)
    sensor_mask[sensor_idx] = 1.0

    m_sensor = m_obs * sensor_mask[:, None, None]
    keep = (rng.random((B, T, 1)) < p_time).astype(np.float32)
    m_train = m_sensor * keep

    return sensor_mask, m_train.astype(np.float32)


class SpatioTemporalMultiContaminantDataset(Dataset):
    """Multi-contaminant spatio-temporal forecasting dataset"""
    def __init__(self, X_dyn, X_static, coords, y_dict, m_train, contaminants, 
                 history_len, horizon, split, train_frac, val_frac, test_frac, seed):
        super().__init__()
        B, T, F = X_dyn.shape
        self.B, self.T, self.F = B, T, F
        self.S = X_static.shape[1]
        self.history_len = history_len
        self.horizon = horizon
        self.contaminants = contaminants

        self.X_dyn = torch.from_numpy(X_dyn).float()
        self.X_static = torch.from_numpy(X_static).float()
        self.coords = torch.from_numpy(coords).float()
        self.y_dict = {c: torch.from_numpy(y_dict[c]).float() for c in contaminants}
        self.m_train = torch.from_numpy(m_train).float()

        t0, t1 = 0, int(T * train_frac)
        t2 = int(T * (train_frac + val_frac))
        
        if split == "train":
            self.t_min, self.t_max = t0, t1
        elif split == "val":
            self.t_min, self.t_max = t1, t2
        else:
            self.t_min, self.t_max = t2, T

        self.valid_times = np.arange(self.t_min, self.t_max)
        self.valid_times = self.valid_times[(self.valid_times >= history_len) & (self.valid_times + horizon < T)]
        self.samples = [(b, int(t)) for b in range(B) for t in self.valid_times]

        rng = np.random.default_rng(seed + hash(split) % 10_000)
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b, t = self.samples[idx]
        L, H = self.history_len, self.horizon

        x_dyn = self.X_dyn[b, t - L : t, :]
        
        y_hist_list = [torch.nan_to_num(self.y_dict[cont][b, t - L : t, :], nan=0.0) for cont in self.contaminants]
        y_hist_filled = torch.cat(y_hist_list, dim=-1)
        m_hist = (torch.isnan(self.y_dict[self.contaminants[0]][b, t - L : t, :]).logical_not()).float()

        y_target_list = [torch.nan_to_num(self.y_dict[cont][b, t + H, :], nan=0.0) for cont in self.contaminants]
        y_target = torch.cat(y_target_list, dim=-1)
        m_t = self.m_train[b, t + H, :]

        seq_in = torch.cat([x_dyn, y_hist_filled, m_hist], dim=-1)

        return {
            "basin_id": torch.tensor(b, dtype=torch.long),
            "coord": self.coords[b, :],
            "x_static": self.X_static[b, :],
            "seq_in": seq_in,
            "y_target": y_target,
            "m_target": m_t,
            "t_index": torch.tensor(t, dtype=torch.long),
        }
