import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd


class ConditionalDenoisingNet(nn.Module):
    def __init__(self, target_len=10, cond_len=30):
        super().__init__()
        self.target_len = target_len
        self.cond_len = cond_len
        self.time_embed = nn.Linear(1, 32)

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_len, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x, t, cond):
        t_emb = F.relu(self.time_embed(t)).unsqueeze(2)
        cond_emb = F.relu(self.cond_encoder(cond.squeeze(1))).unsqueeze(2)
        
        h = F.relu(self.conv1(x))
        h = h + t_emb + cond_emb 
        h = F.relu(self.conv2(h))
        
        h = h.permute(0, 2, 1)
        attn_out, _ = self.attention(h, h, h)
        h = h + attn_out
        
        out = self.output_layer(h)
        out = out.permute(0, 2, 1)
        return out


class ConditionalDiffusion:
    def __init__(self, model, num_timesteps=100):
        self.model = model
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x_start + torch.sqrt(1 - alpha_cumprod_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, cond_data):
        device = next(self.model.parameters()).device
        shape = (cond_data.shape[0], 1, self.model.target_len)
        
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0], 1), i, dtype=torch.float32, device=device) / self.num_timesteps
            
            predicted_noise = self.model(x, t, cond_data)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        return x


print("Downloading AAPL data...")
df = yf.download('AAPL', start='2018-01-01', end='2024-01-01', progress=False)

if isinstance(df.columns, pd.MultiIndex):
    raw_prices = df['Close'].iloc[:, 0].values
else:
    raw_prices = df['Close'].values

cond_len = 30
target_len = 10
total_seq_len = cond_len + target_len

split_idx = int(len(raw_prices) * 0.8)
train_raw = raw_prices[:split_idx]
test_raw = raw_prices[split_idx:]

mean_price = np.mean(train_raw)
std_price = np.std(train_raw)

train_norm = (train_raw - mean_price) / std_price
test_norm = (test_raw - mean_price) / std_price

def create_windows(data, c_len, t_len):
    conds, targets = [], []
    for i in range(len(data) - c_len - t_len):
        conds.append(data[i : i + c_len])
        targets.append(data[i + c_len : i + c_len + t_len])
    return torch.tensor(np.array(conds), dtype=torch.float32).unsqueeze(1), \
           torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)

train_conds, train_targets = create_windows(train_norm, cond_len, target_len)
test_conds, test_targets = create_windows(test_norm, cond_len, target_len)

print(f"Training samples: {len(train_conds)} | Test samples: {len(test_conds)}")

# ==================== اصلاح ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalDenoisingNet(target_len=target_len, cond_len=cond_len).to(device)
diffusion = ConditionalDiffusion(model, num_timesteps=100)

# انتقال تنسورهای داخلی به همان دستگاهی که مدل و داده‌ها روی آن هستند
diffusion.betas = diffusion.betas.to(device)
diffusion.alphas = diffusion.alphas.to(device)
diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
# ==============================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 64
epochs = 100
num_train_samples = len(train_conds)

print("Training started...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    indices = torch.randperm(num_train_samples)
    train_conds = train_conds[indices]
    train_targets = train_targets[indices]
    
    for i in range(0, num_train_samples, batch_size):
        cond_batch = train_conds[i:i+batch_size].to(device)
        target_batch = train_targets[i:i+batch_size].to(device)
        
        t = torch.randint(0, diffusion.num_timesteps, (cond_batch.shape[0],), device=device).long()
        t_float = t.float().unsqueeze(1) / diffusion.num_timesteps
        
        noise = torch.randn_like(target_batch)
        noisy_target = diffusion.q_sample(target_batch, t, noise)
        
        predicted_noise = model(noisy_target, t_float, cond_batch)
        
        loss = F.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/(num_train_samples/batch_size):.6f}")


model.eval()
print("Evaluating on test data...")

test_idx = np.random.randint(0, len(test_conds))
sample_cond = test_conds[test_idx:test_idx+1].to(device)
true_target = test_targets[test_idx:test_idx+1].cpu().squeeze().numpy()

generated_target = diffusion.p_sample_loop(sample_cond).cpu().squeeze().numpy()

sample_cond_real = (sample_cond.cpu().squeeze().numpy() * std_price) + mean_price
true_target_real = (true_target * std_price) + mean_price
generated_target_real = (generated_target * std_price) + mean_price

x_cond = np.arange(cond_len)
x_target = np.arange(cond_len, cond_len + target_len)

plt.figure(figsize=(10, 5))
plt.title("Conditional Diffusion Evaluation on Test Data (Unseen Data)")

plt.plot(x_cond, sample_cond_real, label="Past 30 Days (Condition)", color='blue', linewidth=2)
plt.plot(x_target, true_target_real, label="True Future (10 Days)", color='black', linestyle='-', marker='o')
plt.plot(x_target, generated_target_real, label="Diffusion Prediction", color='red', linestyle='--', marker='x')

plt.axvline(x=cond_len-1, color='gray', linestyle='--', alpha=0.6, label='Prediction Start')
plt.ylabel("Price (USD)")
plt.xlabel("Days")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()