import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_training_history(history, output_dir="figures"):
    """Plot training, validation, and test loss over epochs"""
    os.makedirs(output_dir, exist_ok=True)
    
    df_hist = pd.DataFrame(history)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_hist['epoch'], df_hist['train_loss'], 'o-', linewidth=2, markersize=4, label='Train Loss')
    ax.plot(df_hist['epoch'], df_hist['val_loss'], 's-', linewidth=2, markersize=4, label='Val Loss')
    ax.plot(df_hist['epoch'], df_hist['test_loss'], '^-', linewidth=2, markersize=4, label='Test Loss')
    
    best_epoch = df_hist.loc[df_hist['test_loss'].idxmin(), 'epoch']
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Val (epoch {int(best_epoch)})')
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
    ax.set_title('Training Dynamics: Multi-Contaminant Forecaster', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "01_training_history.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_spatial_maps(y_dict, coords, cfg, output_dir="figures"):
    """Plot spatial contaminant distributions at 3 time points"""
    os.makedirs(output_dir, exist_ok=True)
    
    T = y_dict["no3"].shape[1]
    t_test_start = int(T * (cfg.train_frac + cfg.val_frac))
    t_test_end = T
    
    t_indices = [
        t_test_start + 10,
        t_test_start + (t_test_end - t_test_start) // 2,
        t_test_end - 20,
    ]
    
    num_cont = len(cfg.contaminants)
    fig, axes = plt.subplots(len(t_indices), num_cont, figsize=(14, 4*len(t_indices)))
    if len(t_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, t in enumerate(t_indices):
        for col_idx, cont in enumerate(cfg.contaminants):
            ax = axes[row_idx, col_idx]
            
            y_t = y_dict[cont][:, t, 0]
            
            vmin, vmax = y_t.min(), y_t.max()
            scatter = ax.scatter(coords[:, 1], coords[:, 0], c=y_t, cmap='RdYlGn_r', 
                               s=60, vmin=vmin, vmax=vmax, edgecolors='k', linewidth=0.5)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'{cont.capitalize()} (mg/L)', fontsize=9)
            
            ax.set_xlabel('Longitude', fontsize=9)
            ax.set_ylabel('Latitude', fontsize=9)
            ax.set_title(f'{cont.capitalize()} at Day {t}', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.2)
    
    plt.suptitle('Spatial Contaminant Distributions (Synthetic Data)', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "02_spatial_maps.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_forecast_trajectories(model, ds_test, y_dict, cfg, device, output_dir="figures"):
    """Plot forecast vs observations for 3 selected basins"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select 3 random basins from test set
    unique_basins = list(set([b for b, t in ds_test.samples]))
    if len(unique_basins) > 3:
        test_basin_indices = list(np.random.choice(unique_basins, size=3, replace=False))
    else:
        test_basin_indices = unique_basins[:3]
    
    num_cont = len(cfg.contaminants)
    fig, axes = plt.subplots(len(test_basin_indices), num_cont, figsize=(14, 4*len(test_basin_indices)))
    if len(test_basin_indices) == 1:
        axes = axes.reshape(1, -1)
    
    model.eval()
    
    for row_idx, b_idx in enumerate(test_basin_indices):
        basin_samples = [i for i, (b, t) in enumerate(ds_test.samples) if b == b_idx]
        
        if len(basin_samples) == 0:
            continue
        
        step = max(1, len(basin_samples) // 10)
        selected_samples = basin_samples[::step][:10]
        
        for col_idx, cont in enumerate(cfg.contaminants):
            ax = axes[row_idx, col_idx]
            
            y_true_list = []
            y_pred_list = []
            t_list = []
            
            for sample_idx in selected_samples:
                batch = ds_test[sample_idx]
                
                y_true = batch['y_target'].cpu().numpy()[col_idx]
                t_idx = batch['t_index'].cpu().numpy()
                
                with torch.no_grad():
                    seq_in_t = batch['seq_in'].float().unsqueeze(0).to(device)
                    basin_id_t = batch['basin_id'].unsqueeze(0).to(device)
                    x_static_t = batch['x_static'].float().unsqueeze(0).to(device)
                    coord_t = batch['coord'].float().unsqueeze(0).to(device)
                    
                    y_hat = model(basin_id_t, x_static_t, coord_t, seq_in_t)
                    y_hat_val = y_hat.squeeze(0).cpu().numpy()[col_idx]
                
                y_true_list.append(float(y_true))
                y_pred_list.append(float(y_hat_val))
                t_list.append(int(t_idx))
            
            ax.plot(t_list, y_true_list, 'o-', color='navy', linewidth=2, markersize=6, label='Observed')
            ax.plot(t_list, y_pred_list, 's--', color='red', linewidth=2, markersize=5, label='Forecast')
            ax.set_xlabel('Time (days)', fontsize=9)
            ax.set_ylabel(f'{cont.capitalize()} (mg/L)', fontsize=9)
            ax.set_title(f'Basin {b_idx:03d} - {cont.capitalize()}', fontsize=10, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.2)
    
    plt.suptitle(f'7-Day Ahead Forecasts (Test Period)', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "03_forecast_trajectories.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_parity_diagrams(y_true, y_pred, m, cfg, output_dir="figures"):
    """Plot predicted vs observed (parity plots)"""
    os.makedirs(output_dir, exist_ok=True)
    
    from training import rmse, mae
    
    fig, axes = plt.subplots(1, len(cfg.contaminants), figsize=(5*len(cfg.contaminants), 4))
    if len(cfg.contaminants) == 1:
        axes = [axes]
    
    idx = m[:, 0] > 0.5
    
    for i, cont in enumerate(cfg.contaminants):
        ax = axes[i]
        
        y_t_c = y_true[idx, i]
        y_p_c = y_pred[idx, i]
        
        ax.scatter(y_t_c, y_p_c, alpha=0.4, s=20, edgecolors='none')
        
        lims = [min(y_t_c.min(), y_p_c.min()), max(y_t_c.max(), y_p_c.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect Forecast')
        
        rmse_c = rmse(y_t_c, y_p_c)
        r2 = 1 - np.sum((y_t_c - y_p_c)**2) / np.sum((y_t_c - y_t_c.mean())**2)
        
        ax.set_xlabel(f'Observed {cont.capitalize()} (mg/L)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Forecast {cont.capitalize()} (mg/L)', fontsize=10, fontweight='bold')
        ax.set_title(f'{cont.capitalize()}\nRMSE={rmse_c:.4f}, R²={r2:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle('Parity Plots: Test Set Predictions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, "04_parity_diagrams.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


def plot_metric_summary(y_true, y_pred, m, cfg, output_dir="figures"):
    """Create a summary table of metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    from training import rmse, mae, nrmse
    
    idx = m[:, 0] > 0.5
    
    metrics_data = []
    for i, cont in enumerate(cfg.contaminants):
        y_t = y_true[idx, i]
        y_p = y_pred[idx, i]
        metrics_data.append({
            'Contaminant': cont.upper(),
            'RMSE': f"{rmse(y_t, y_p):.4f}",
            'MAE': f"{mae(y_t, y_p):.4f}",
            'NRMSE': f"{nrmse(y_t, y_p):.4f}",
            'Points': idx.sum()
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_metrics.values, colLabels=df_metrics.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df_metrics.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df_metrics) + 1):
        for j in range(len(df_metrics.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Test Set Metrics Summary', fontsize=12, fontweight='bold', pad=20)
    
    path = os.path.join(output_dir, "05_metrics_table.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()
    
    return df_metrics


def save_all_plots(model, history, y_true, y_pred, m, ds_test, y_dict, coords, cfg, device, output_dir="figures"):
    """Generate and save all plots"""
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    plot_training_history(history, output_dir)
    plot_spatial_maps(y_dict, coords, cfg, output_dir)
    plot_forecast_trajectories(model, ds_test, y_dict, cfg, device, output_dir)
    plot_parity_diagrams(y_true, y_pred, m, cfg, output_dir)
    df_metrics = plot_metric_summary(y_true, y_pred, m, cfg, output_dir)
    
    print("="*70)
    print(f"All figures saved to '{output_dir}/'")
    print("="*70)
    
    return df_metrics