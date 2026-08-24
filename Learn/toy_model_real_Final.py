import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Loading nasdq.csv dataset...")
df = pd.read_csv('/home/alien-hawk/Desktop/PEDC/Learn/nasdq.csv')

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'InterestRate', 'ExchangeRate', 'VIX', 'TEDSpread', 'EFFR', 'Gold', 'Oil']
data_values = df[features].values
close_idx = features.index('Close')

cond_len = 30   
target_len = 10 
N_stocks = 3    

def create_graph_dataset_windowed(data, c_len, t_len, n_nodes, target_col_idx):
    conds, targets = [], []
    means, stds = [], []
    
    for i in range(len(data) - c_len - t_len):
        c_window = data[i : i + c_len].copy()
        t_window = data[i + c_len : i + c_len + t_len, target_col_idx:target_col_idx+1].copy()
        
        w_mean = np.mean(c_window, axis=0)
        w_std = np.std(c_window, axis=0) + 1e-8
        
        c_norm = (c_window - w_mean) / w_std
        t_norm = (t_window - w_mean[target_col_idx]) / w_std[target_col_idx]
        
        # افزودن نویز جزئی برای ایجاد چند سهام متفاوت (Data Augmentation)
        c_nodes = [c_norm + np.random.normal(0, 0.01, c_norm.shape) for _ in range(n_nodes)]
        t_nodes = [t_norm + np.random.normal(0, 0.01, t_norm.shape) for _ in range(n_nodes)]
        
        conds.append(np.array(c_nodes).transpose(0, 2, 1))
        targets.append(np.array(t_nodes).transpose(0, 2, 1))
        
        means.append(w_mean[target_col_idx])
        stds.append(w_std[target_col_idx])
        
    return torch.tensor(np.array(conds), dtype=torch.float32), \
           torch.tensor(np.array(targets), dtype=torch.float32), \
           np.array(means), np.array(stds)

split_idx = int(len(data_values) * 0.8)
train_data = data_values[:split_idx]
test_data = data_values[split_idx:]

train_conds, train_targets, _, _ = create_graph_dataset_windowed(train_data, cond_len, target_len, N_stocks, close_idx)
test_conds, test_targets, test_means, test_stds = create_graph_dataset_windowed(test_data, cond_len, target_len, N_stocks, close_idx)

relation_matrix = torch.ones((N_stocks, N_stocks))

class AttDiCEm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        return F.relu(self.conv(x))

class MaskedRelationalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, relation_matrix):
        B, N, _ = x.shape
        mask = torch.where(relation_matrix == 0, float('-inf'), 0.0).to(x.device)
        mask = mask.repeat(B * self.num_heads, 1, 1)
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)

class DiffStockMaTCHS(nn.Module):
    def __init__(self, d_channels=32):
        super().__init__()
        self.d = d_channels
        self.time_embed = nn.Linear(1, self.d)
        
        self.cnn_cond = AttDiCEm(in_channels=12, out_channels=self.d)
        self.cnn_target = AttDiCEm(in_channels=1, out_channels=self.d)
        
        self.total_time = cond_len + target_len
        self.embed_dim = self.d * self.total_time
        self.mrt = MaskedRelationalTransformer(embed_dim=self.embed_dim, num_heads=4)
        
        self.cnn_final = AttDiCEm(in_channels=self.d, out_channels=self.d)
        self.output_layer = nn.Linear(self.d, 1)

    def forward(self, noisy_target, t, cond_data, rel_matrix):
        B, N, _, _ = cond_data.shape
        c_flat = cond_data.view(B * N, 12, cond_len)
        t_flat = noisy_target.view(B * N, 1, target_len)
        
        h_cond = self.cnn_cond(c_flat)
        h_target = self.cnn_target(t_flat)
        
        h = torch.cat([h_cond, h_target], dim=2)
        t_emb = F.relu(self.time_embed(t)).repeat_interleave(N, dim=0).unsqueeze(2)
        h = h + t_emb
        
        h = h.view(B, N, self.embed_dim)
        h = self.mrt(h, rel_matrix)
        
        h = h.view(B * N, self.d, self.total_time)
        h = self.cnn_final(h)
        h = h.permute(0, 2, 1)
        out = self.output_layer(h)
        out = out.permute(0, 2, 1).view(B, N, 1, self.total_time)
        return out[:, :, :, -target_len:]

class Diffusion:
    def __init__(self, model, num_timesteps=100):
        self.model = model
        self.num_timesteps = num_timesteps
        device = next(model.parameters()).device  # دریافت دستگاه از مدل
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None: 
            noise = torch.randn_like(x_start)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x_start + torch.sqrt(1 - alpha_cumprod_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, cond_data, rel_matrix):
        device = next(self.model.parameters()).device
        B, N, _, _ = cond_data.shape
        shape = (B, N, 1, target_len)
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((B, 1), i, dtype=torch.float32, device=device) / self.num_timesteps
            predicted_noise = self.model(x, t, cond_data, rel_matrix)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            if i > 0: 
                noise = torch.randn_like(x)
            else: 
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        return x

# ==================== تنظیم دستگاه ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
relation_matrix = relation_matrix.to(device)

model = DiffStockMaTCHS(d_channels=32).to(device)
diffusion = Diffusion(model, num_timesteps=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 64
epochs = 80
num_train_samples = len(train_conds)

# ==================== حلقه آموزش ====================
print(f"\nStarting network training for {epochs} epochs...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    indices = torch.randperm(num_train_samples)
    for i in range(0, num_train_samples, batch_size):
        idx = indices[i:i+batch_size]
        cond_batch = train_conds[idx].to(device)
        target_batch = train_targets[idx].to(device)
        
        t = torch.randint(0, diffusion.num_timesteps, (cond_batch.shape[0],), device=device).long()
        t_float = t.float().unsqueeze(1) / diffusion.num_timesteps
        
        noise = torch.randn_like(target_batch)
        noisy_target = diffusion.q_sample(target_batch, t, noise)
        predicted_noise = model(noisy_target, t_float, cond_batch, relation_matrix)
        
        loss = F.mse_loss(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/(num_train_samples/batch_size):.6f}")

# ==================== ارزیابی و رسم ====================
model.eval()
print("\nGenerating prediction...")

test_idx = np.random.randint(0, len(test_conds))
sample_cond = test_conds[test_idx:test_idx+1].to(device) 
true_target = test_targets[test_idx:test_idx+1].cpu().numpy() 

generated_target = diffusion.p_sample_loop(sample_cond, relation_matrix).cpu().numpy()

window_mean = test_means[test_idx]
window_std = test_stds[test_idx]

past_real = (sample_cond.cpu().numpy()[0, 0, close_idx, :] * window_std) + window_mean
true_future_real = (true_target[0, 0, 0, :] * window_std) + window_mean
pred_future_real = (generated_target[0, 0, 0, :] * window_std) + window_mean

x_cond = np.arange(cond_len)
x_target = np.arange(cond_len, cond_len + target_len)

plt.figure(figsize=(10, 5))
plt.title("DiffStock Prediction (Window Normalized + Higher Epochs)")

plt.plot(x_cond, past_real, label="Past 30 Days", color='blue', linewidth=2)
plt.plot(x_target, true_future_real, label="True Future", color='black', marker='o')
plt.plot(x_target, pred_future_real, label="Diffusion Prediction", color='red', linestyle='--', marker='x')

plt.axvline(x=cond_len-1, color='gray', linestyle='--')
plt.ylabel("Close Price")
plt.xlabel("Days")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()