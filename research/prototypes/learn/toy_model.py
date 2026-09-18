import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output + x


class ToyDenoisingNet(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.time_embed = nn.Linear(1, 16)
        
        self.cnn = SimpleCNN1D(in_channels=1, out_channels=16)
        self.attention = SimpleAttention(embed_dim=16, num_heads=2)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x, t):
        t_emb = F.relu(self.time_embed(t)).unsqueeze(2)
        
        out = self.cnn(x)
        out = out + t_emb
        
        out = out.permute(0, 2, 1)
        
        out = self.attention(out)
        
        out = self.output_layer(out)
        out = out.permute(0, 2, 1)
        return out


class ToyDiffusion:
    def __init__(self, model, num_timesteps=100):
        self.model = model
        self.num_timesteps = num_timesteps
        # دریافت دستگاه از مدل (که قبلاً به device منتقل شده)
        device = next(model.parameters()).device
        # ایجاد تنسورها و انتقال مستقیم به همان دستگاه
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas  # خودکار روی device می‌ماند
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        return torch.sqrt(alpha_cumprod_t) * x_start + torch.sqrt(1 - alpha_cumprod_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.model.parameters()).device
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0], 1), i, dtype=torch.float32, device=device) / self.num_timesteps
            predicted_noise = self.model(x, t)
            
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
            
        return x


# ==================== ساخت داده ====================
seq_length = 50
num_samples = 1000
x_data = np.linspace(0, 10, seq_length)
clean_signal = np.sin(x_data)
dataset = [clean_signal + np.random.normal(0, 0.1, seq_length) for _ in range(num_samples)]
dataset = torch.tensor(np.array(dataset), dtype=torch.float32).unsqueeze(1)

# ==================== تنظیم دستگاه ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoising_net = ToyDenoisingNet(seq_len=seq_length).to(device)
diffusion = ToyDiffusion(denoising_net, num_timesteps=100)

optimizer = torch.optim.Adam(denoising_net.parameters(), lr=1e-3)
batch_size = 64
epochs = 50

print("Training started... (this will take a few seconds)")
denoising_net.train()
for epoch in range(epochs):
    epoch_loss = 0
    # shuffle data each epoch
    indices = torch.randperm(num_samples)
    dataset_shuffled = dataset[indices]
    
    for i in range(0, num_samples, batch_size):
        batch = dataset_shuffled[i:i+batch_size].to(device)
        
        t = torch.randint(0, diffusion.num_timesteps, (batch.shape[0],), device=device).long()
        t_float = t.float().unsqueeze(1) / diffusion.num_timesteps
        
        noise = torch.randn_like(batch)
        
        noisy_batch = diffusion.q_sample(batch, t, noise)
        
        predicted_noise = denoising_net(noisy_batch, t_float)
        
        loss = F.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch+1) % 10 == 0:
        # میانگین损失 در هر نمونه (تعداد کل نمونه‌ها)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / num_samples:.6f}")

# ==================== تولید نمونه ====================
denoising_net.eval()
print("Generating new data using Diffusion...")
generated_sample = diffusion.p_sample_loop(shape=(1, 1, seq_length)).cpu().squeeze().numpy()


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Real Data Sample (Training)")
plt.plot(dataset[0].squeeze().numpy(), label="Real Data (Stock Trend)", color='blue')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Generated Output from Diffusion Model")
plt.plot(generated_sample, label="Generated Diffusion Output", color='green', linestyle='--')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()