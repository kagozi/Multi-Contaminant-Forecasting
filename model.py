import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple 2-layer MLP"""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SpatialGRUMultiContaminantForecaster(nn.Module):
    """Spatial embedding + GRU encoder with multi-contaminant heads"""
    def __init__(self, num_basins: int, f_in: int, static_dim: int, num_contaminants: int,
                 d_basin: int = 32, d_static: int = 32, d_coord: int = 16, d_model: int = 64,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_contaminants = num_contaminants

        # Spatial embeddings
        self.emb_basin = nn.Embedding(num_basins, d_basin)
        self.proj_static = MLP(static_dim, max(16, d_static), d_static, dropout=dropout)
        self.proj_coord = MLP(2, max(8, d_coord), d_coord, dropout=dropout)

        ctx_dim = d_basin + d_static + d_coord
        self.ctx_to_h0 = nn.Sequential(
            nn.Linear(ctx_dim, d_model * num_layers),
            nn.Tanh(),
        )

        # Shared GRU
        self.gru = nn.GRU(
            input_size=f_in,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Contaminant-specific heads
        self.heads = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            for i in range(num_contaminants)
        })

    def forward(self, basin_id, x_static, coord, seq_in):
        eb = self.emb_basin(basin_id)
        es = self.proj_static(x_static)
        ec = self.proj_coord(coord)
        ctx = torch.cat([eb, es, ec], dim=-1)

        h0_flat = self.ctx_to_h0(ctx)
        h0 = h0_flat.view(-1, self.num_layers, self.d_model).transpose(0, 1).contiguous()

        out, _ = self.gru(seq_in, h0)
        h_last = out[:, -1, :]

        y_hat = torch.cat([self.heads[str(i)](h_last) for i in range(self.num_contaminants)], dim=-1)
        return y_hat
