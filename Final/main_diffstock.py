import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class MultiMarketStockDataset(Dataset):
    def __init__(self, data_dir, markets=["nasdaq", "nyse", "sp500", "forbes2000"], 
                 seq_len_past=30, seq_len_future=10, max_stocks=100, corr_threshold=0.5):
        self.seq_len_past = seq_len_past
        self.seq_len_future = seq_len_future
        self.total_seq_len = seq_len_past + seq_len_future
        
        all_csv_files = []
        for market in markets:
            csv_path = os.path.join(data_dir, market, "csv")
            if os.path.exists(csv_path):
                files = glob.glob(os.path.join(csv_path, "*.csv")) + glob.glob(os.path.join(csv_path, "*.CSV"))
                all_csv_files.extend(files)
        
        all_csv_files = sorted(list(set(all_csv_files)))
        
        if max_stocks is not None:
            all_csv_files = all_csv_files[:max_stocks]
            
        print(f"Scanning {len(all_csv_files)} total CSV files...")
        
        dataframes = {}
        close_prices = {}
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for file in all_csv_files:
            stock_name = os.path.basename(file).replace(".csv", "").replace(".CSV", "")
            try:
                df = pd.read_csv(file, on_bad_lines='skip')
                date_col = next((col for col in ['Date', 'date', 'Timestamp', 'time'] if col in df.columns), None)
                if date_col is None: continue
                
                df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                df.dropna(subset=['Date'], inplace=True)
                df.set_index('Date', inplace=True)
                df = df.reindex(columns=features)
                
                dataframes[stock_name] = df
                close_prices[stock_name] = df['Close']
            except Exception:
                pass
        
        print("Merging all market datasets...")
        merged_df = pd.concat(dataframes, axis=1, join='outer')
        merged_df.ffill(inplace=True)
        merged_df.bfill(inplace=True)
        merged_df.fillna(0.0, inplace=True)
        
        self.raw_data = merged_df.values
        self.num_stocks = len(dataframes)
        self.num_features = len(features)
        
        # 🟢 بُعد دادن به داده‌ها (اینجا درست شد)
        self.raw_data = self.raw_data.reshape(-1, self.num_stocks, self.num_features)
        
        close_df = pd.DataFrame(close_prices)
        close_df.ffill(inplace=True)
        close_df.bfill(inplace=True)
        
        returns_df = close_df.pct_change().fillna(0.0)
        corr_matrix = np.nan_to_num(returns_df.corr().values, nan=0.0)
        
        self.relation_matrix = np.zeros((self.num_stocks, self.num_stocks, 2), dtype=np.float32)
        self.relation_matrix[:, :, 0] = (corr_matrix > corr_threshold).astype(np.float32)
        self.relation_matrix[:, :, 1] = (corr_matrix < -corr_threshold).astype(np.float32)
        
        np.fill_diagonal(self.relation_matrix[:, :, 0], 1.0)
        np.fill_diagonal(self.relation_matrix[:, :, 1], 1.0)
        
        self.relation_matrix = torch.tensor(self.relation_matrix, dtype=torch.float32)
        self.valid_indices = len(self.raw_data) - self.total_seq_len

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        window = self.raw_data[idx : idx + self.total_seq_len]
        
        past_raw = window[:self.seq_len_past]      
        future_raw = window[self.seq_len_past:]    
        
        mean = np.mean(past_raw, axis=0, keepdims=True)
        std = np.std(past_raw, axis=0, keepdims=True) + 1e-5
        
        past_norm = (past_raw - mean) / std
        future_norm = (future_raw - mean) / std
        
        return (torch.tensor(past_norm, dtype=torch.float32).permute(1, 2, 0),
                torch.tensor(future_norm, dtype=torch.float32).permute(1, 2, 0),
                self.relation_matrix)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class AttDiCEm(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=self.padding, dilation=dilation)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        return self.gelu(x[:, :, :-self.padding])

class MaskedRelationalTransformer(nn.Module):
    def __init__(self, embed_dim, num_masked_heads=2, num_unmasked_heads=2):
        super().__init__()
        self.total_heads = num_masked_heads + num_unmasked_heads
        self.embed_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_masked = num_masked_heads

    def forward(self, x, relation_matrix):
        B, N, D = x.shape
        H = self.total_heads
        head_dim = D // H

        qkv = self.qkv(x).reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        if relation_matrix is not None:
            mask = torch.full((H, N, N), -1e9, device=x.device)
            G = relation_matrix.shape[-1]
            for g in range(min(G, self.num_masked)):
                rel_mask = relation_matrix[:, :, g] > 0
                mask[g][rel_mask] = 0.0
            mask[self.num_masked:] = 0.0 
            scores = scores + mask.unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.norm(x + self.proj(out))

class MaTCHS(nn.Module):
    def __init__(self, num_features=5, d_channels=64, seq_len=40):
        super().__init__()
        self.d = d_channels
        self.seq_len = seq_len
        self.noise_encoder = AttDiCEm(num_features, self.d, dilation=1)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(self.d), nn.Linear(self.d, self.d * 4), nn.GELU(), nn.Linear(self.d * 4, self.d))
        self.mrt = MaskedRelationalTransformer(embed_dim=self.d * self.seq_len)
        self.final_conv = nn.Sequential(AttDiCEm(self.d, self.d), nn.Conv1d(self.d, num_features, kernel_size=1))

    def forward(self, noisy_target, t, condition, rel_matrix):
        B, N, P, _ = condition.shape
        _, _, _, L_fut = noisy_target.shape
        x_concat = torch.cat([condition, noisy_target], dim=-1)
        L_total = x_concat.shape[-1]
        
        x_flat = x_concat.view(B * N, P, L_total)
        h = self.noise_encoder(x_flat)
        
        t_emb = self.time_mlp(t).repeat_interleave(N, dim=0).unsqueeze(-1)
        h = h + t_emb
        
        h_spatial = h.view(B, N, self.d * L_total)
        h_spatial = self.mrt(h_spatial, rel_matrix)
        
        out = self.final_conv(h_spatial.view(B * N, self.d, L_total)).view(B, N, P, L_total)
        return out[:, :, :, -L_fut:]

class AdaptiveStockDiffusion:
    def __init__(self, model, num_timesteps=100, device='cuda'):
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.2, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x_start + torch.sqrt(1 - alpha_cumprod_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, condition, rel_matrix, future_len=10):
        B, N, P, _ = condition.shape
        shape = (B, N, P, future_len)
        x = torch.randn(shape, device=self.device)
        
        for i in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((B,), i, dtype=torch.float32, device=self.device) / self.num_timesteps
            predicted_noise = self.model(x, t_tensor, condition, rel_matrix)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using AI device: {device}")
    
    base_dir = "/home/alien-hawk/Desktop/PEDC/Final/stock_market_data"
    
    dataset = MultiMarketStockDataset(data_dir=base_dir, max_stocks=100, corr_threshold=0.5)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    
    denoising_net = MaTCHS(num_features=5, d_channels=64, seq_len=40)
    diffusion = AdaptiveStockDiffusion(denoising_net, num_timesteps=100, device=device)
    optimizer = torch.optim.AdamW(denoising_net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 30
    print(f"\nStarting Training of DiffStock on {dataset.num_stocks} Stocks...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        denoising_net.train()
        
        for batch_idx, (past, future, rel_matrix) in enumerate(dataloader):
            past = past.to(device)
            future = future.to(device)
            static_rel_matrix = rel_matrix[0].to(device) 
            
            t = torch.randint(0, diffusion.num_timesteps, (future.shape[0],), device=device).long()
            noise = torch.randn_like(future)
            noisy_future = diffusion.q_sample(future, t, noise)
            
            pred_noise = denoising_net(noisy_future, t, past, static_rel_matrix)
            loss = F.mse_loss(pred_noise, noise)
            
            if torch.isnan(loss): continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoising_net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch + 1:02d}/{epochs}] | Average MSE Loss: {avg_loss:.6f}")
            
    print("\nSaving Model Weights First...")
    torch.save(denoising_net.state_dict(), 'matches_diffusion_weights.pth')
    
    print("\nFinding an active and highly volatile trading window...")
    best_idx = 0
    max_volatility = -1
    search_start = max(0, dataset.valid_indices - 2000)
    
    for i in range(search_start, dataset.valid_indices):
        window_raw = dataset.raw_data[i : i + 30, :, 3] 
        mean_prices = np.mean(window_raw, axis=0) + 1e-5
        std_prices = np.std(window_raw, axis=0)
        cv = np.mean(std_prices / mean_prices)
        if cv > max_volatility:
            max_volatility = cv
            best_idx = i

    print(f"Selected active window starting at index {best_idx}")
    
    window_raw = dataset.raw_data[best_idx : best_idx + dataset.total_seq_len]
    past_raw = window_raw[:dataset.seq_len_past]
    future_raw = window_raw[dataset.seq_len_past:]
    
    mean_val = np.mean(past_raw, axis=0, keepdims=True)
    std_val = np.std(past_raw, axis=0, keepdims=True) + 1e-5
    past_norm = (past_raw - mean_val) / std_val
    
    sample_past = torch.tensor(past_norm, dtype=torch.float32).permute(1, 2, 0).unsqueeze(0).to(device)
    static_rel = dataset.relation_matrix.to(device)
    
    denoising_net.eval()
    num_samples = 10
    all_predictions = []
    
    for _ in range(num_samples):
        pred_norm = diffusion.p_sample_loop(sample_past, static_rel, future_len=10)
        all_predictions.append(pred_norm.cpu().numpy())
        
    all_predictions = np.stack(all_predictions)
    
    feature_idx = 3
    
    cv_per_stock = np.std(past_raw[:, :, feature_idx], axis=0) / (np.mean(past_raw[:, :, feature_idx], axis=0) + 1e-5)
    best_stocks_to_plot = np.argsort(cv_per_stock)[-4:][::-1]
    
    x_past = np.arange(30)
    x_future = np.arange(30, 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("DiffStock (MaTCHS) Predictions - Most Recent & Volatile Stocks", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for i, stock_idx in enumerate(best_stocks_to_plot):
        ax = axes[i]
        
        m_v = mean_val[0, stock_idx, feature_idx]
        s_v = std_val[0, stock_idx, feature_idx]
        
        p_prices = past_raw[:, stock_idx, feature_idx]
        f_prices = future_raw[:, stock_idx, feature_idx]
        
        ax.plot(x_past, p_prices, label="Past 30 Days", color="blue", linewidth=2.5)
        ax.plot(x_future, f_prices, label="True Future", color="black", marker="o", linewidth=2.5)
        
        for s in range(num_samples):
            s_pred_norm = all_predictions[s, 0, stock_idx, feature_idx, :]
            s_pred = (s_pred_norm * s_v) + m_v
            ax.plot(x_future, s_pred, color="red", alpha=0.15)
            
        mean_pred_norm = all_predictions.mean(axis=0)[0, stock_idx, feature_idx, :]
        mean_pred = (mean_pred_norm * s_v) + m_v
        ax.plot(x_future, mean_pred, label="Mean Prediction", color="red", marker="x", linestyle="--", linewidth=2.5)
        
        ax.axvline(x=29, color="gray", linestyle="--")
        ax.set_title(f"Stock Index: {stock_idx}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Close Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("diffstock_realistic_predictions.png", dpi=300)
    print("\nPlot saved as 'diffstock_realistic_predictions.png'!")
    plt.show()