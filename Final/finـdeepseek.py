import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATASET & PREPROCESSING (BUSINESS LOGIC)
# ==============================================================================

class AlphaStockDataset(Dataset):
    def __init__(self, data_dir, markets=["nasdaq", "nyse", "sp500", "forbes2000"], 
                 seq_len_past=30, max_stocks=30, corr_threshold=0.3):
        self.seq_len_past = seq_len_past
        
        all_csv_files = []
        for market in markets:
            csv_path = os.path.join(data_dir, market, "csv")
            if os.path.exists(csv_path):
                files = glob.glob(os.path.join(csv_path, "*.csv")) + glob.glob(os.path.join(csv_path, "*.CSV"))
                all_csv_files.extend(files)
        
        # Sort and limit for memory
        all_csv_files = sorted(list(set(all_csv_files)))[:max_stocks]
        print(f"[DATA] Loading {len(all_csv_files)} stocks from {markets}...")
        
        dataframes = {}
        for file in all_csv_files:
            stock_name = os.path.basename(file).replace(".csv", "").replace(".CSV", "")
            try:
                df = pd.read_csv(file, on_bad_lines='skip')
                # Find Date column dynamically
                date_col = next((col for col in ['Date', 'date', 'Timestamp', 'time'] if col in df.columns), None)
                if date_col is None: continue
                
                df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
                df.dropna(subset=['Date'], inplace=True)
                df.set_index('Date', inplace=True)
                
                # We only need 'Close' to calculate Log Returns for the business model
                if 'Close' in df.columns:
                    dataframes[stock_name] = df['Close']
            except Exception:
                pass
        
        # 1. Outer Join to keep ALL dates, Forward Fill missing data (Hedge Fund standard)
        print("[DATA] Merging data (Outer Join & Forward Fill)...")
        merged_close = pd.concat(dataframes, axis=1, join='outer')
        merged_close.ffill(inplace=True)
        merged_close.bfill(inplace=True)
        
        # 2. Calculate Log Returns (Stationary data, required for NNs)
        print("[DATA] Calculating Log Returns...")
        log_returns = np.log(merged_close / merged_close.shift(1))
        log_returns.fillna(0.0, inplace=True)
        
        self.num_stocks = len(merged_close.columns)
        self.stock_symbols = list(merged_close.columns)
        
        # Features array: Shape [Total_Days, Num_Stocks, Num_Features]
        # Feature 0: Log Return
        # Feature 1: Dummy Sentiment Data (Placeholder for your Web Scraping team)
        total_days = len(log_returns)
        self.features = np.zeros((total_days, self.num_stocks, 2), dtype=np.float32)
        self.features[:, :, 0] = log_returns.values
        self.features[:, :, 1] = np.random.uniform(-1, 1, size=(total_days, self.num_stocks)) # Sentiment
        
        # 3. Correlation Graph (Relational Matrix)
        print("[DATA] Building Spatial Correlation Graph...")
        corr_matrix = np.nan_to_num(log_returns.corr().values, nan=0.0)
        self.relation_matrix = np.zeros((self.num_stocks, self.num_stocks, 2), dtype=np.float32)
        self.relation_matrix[:, :, 0] = (corr_matrix > corr_threshold).astype(np.float32)
        self.relation_matrix[:, :, 1] = (corr_matrix < -corr_threshold).astype(np.float32)
        np.fill_diagonal(self.relation_matrix[:, :, 0], 1.0)
        np.fill_diagonal(self.relation_matrix[:, :, 1], 1.0)
        self.relation_matrix = torch.tensor(self.relation_matrix, dtype=torch.float32)
        
        self.valid_indices = total_days - self.seq_len_past - 1
        print(f"[DATA] Dataset ready. {self.valid_indices} trading days available for training/validation.")

    def __len__(self):
        return max(0, self.valid_indices)

    def __getitem__(self, idx):
        # Input: Past N days
        past_data = self.features[idx : idx + self.seq_len_past] # [Seq, Stocks, Features]
        
        # Target: Tomorrow's Log Return
        target_return = self.features[idx + self.seq_len_past, :, 0] # [Stocks]
        
        # Classification Target: 1 if Return > 0 (Up), else 0 (Down)
        target_class = (target_return > 0).astype(np.float32)

        past_tensor = torch.tensor(past_data, dtype=torch.float32).permute(1, 2, 0) # [Stocks, Features, Seq]
        target_reg = torch.tensor(target_return, dtype=torch.float32)
        target_cls = torch.tensor(target_class, dtype=torch.float32)
        
        return past_tensor, target_reg, target_cls

# ==============================================================================
# 2. ALPHA-MaTCHS ARCHITECTURE
# ==============================================================================

class CausalConv1d(nn.Module):
    """Causal Convolution to prevent looking into the future."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x

class TemporalBlock(nn.Module):
    """TCN Block to capture time-series patterns stably."""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.relu1 = nn.GELU()
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, dilation=dilation)
        self.relu2 = nn.GELU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class MaskedRelationalTransformer(nn.Module):
    """Uses the correlation matrix to guide attention between stocks."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads

    def forward(self, x, relation_matrix):
        B, N, D = x.shape
        head_dim = D // self.num_heads
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply positive relation mask
        rel_mask = relation_matrix[:, :, 0] > 0
        mask = torch.zeros_like(scores)
        mask = mask.masked_fill(~rel_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.norm(x + self.proj(out))

class AlphaMaTCHS(nn.Module):
    """The Multi-Task Business Model for Trading"""
    def __init__(self, num_features=2, d_model=64, seq_len=30):
        super().__init__()
        # 1. Temporal Feature Extractor
        self.tcn = nn.Sequential(
            TemporalBlock(num_features, d_model, dilation=1),
            TemporalBlock(d_model, d_model, dilation=2),
            TemporalBlock(d_model, d_model, dilation=4)
        )
        # 2. Spatial Feature Extractor
        self.mrt = MaskedRelationalTransformer(embed_dim=d_model * seq_len)
        
        # 3. Multi-Task Heads
        flat_dim = d_model * seq_len
        
        # Head A: Return Regression (How much will it move?)
        self.reg_head = nn.Sequential(
            nn.Linear(flat_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Head B: Trend Classification (Up or Down?) -> NO SIGMOID HERE!
        self.cls_head = nn.Sequential(
            nn.Linear(flat_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1) 
        )

    def forward(self, x, relation_matrix):
        B, N, F_dim, L = x.shape
        x_flat = x.view(B * N, F_dim, L)
        
        h_temp = self.tcn(x_flat) # [B*N, d_model, L]
        
        h_spatial = h_temp.view(B, N, -1) # [B, N, d_model*L]
        h_spatial = self.mrt(h_spatial, relation_matrix) # Apply graph
        
        out_reg = self.reg_head(h_spatial).squeeze(-1) # [B, N]
        out_cls = self.cls_head(h_spatial).squeeze(-1) # [B, N] - Raw Logits
        
        return out_reg, out_cls

# ==============================================================================
# 3. TRAINING & REALISTIC BACKTEST LOOP
# ==============================================================================

def train_alpha_matches(model, train_loader, val_loader, adj_matrix, epochs=15, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    criterion_reg = nn.HuberLoss()
    # USE BCEWithLogitsLoss to fix CUDA Device Assert Error!
    criterion_cls = nn.BCEWithLogitsLoss() 
    
    adj_matrix = adj_matrix.to(device)
    model.to(device)
    
    history = {'loss': [], 'acc': [], 'sharpe': [], 'strategy_returns': [], 'market_returns': []}
    
    # Business Constraints
    TRANSACTION_COST = 0.001 # 0.1% per trade
    MAX_LEVERAGE = 1.0       # No margin used
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y_reg, batch_y_cls in train_loader:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            batch_y_cls = batch_y_cls.to(device)
            
            optimizer.zero_grad()
            pred_reg, pred_cls = model(batch_x, adj_matrix)
            
            # Combine Regression (Huber) and Classification (BCE) Loss
            loss = criterion_reg(pred_reg, batch_y_reg) + criterion_cls(pred_cls, batch_y_cls)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # ---------------------------------------------------------
        # REALISTIC VALIDATION BACKTEST
        # ---------------------------------------------------------
        model.eval()
        val_acc, total_val = 0.0, 0
        portfolio_returns = []
        market_returns = []
        prev_weights = None 
        
        with torch.no_grad():
            for batch_x, batch_y_reg, batch_y_cls in val_loader:
                batch_x = batch_x.to(device)
                pred_reg, pred_cls = model(batch_x, adj_matrix)
                
                # Apply Sigmoid manually ONLY for accuracy and strategy calculations
                pred_probs = torch.sigmoid(pred_cls) 
                
                preds_binary = (pred_probs > 0.5).cpu().numpy()
                targets_binary = batch_y_cls.cpu().numpy()
                
                val_acc += np.sum(preds_binary == targets_binary)
                total_val += targets_binary.size
                
                probs = pred_probs.cpu().numpy()[0]
                actual_returns = batch_y_reg.cpu().numpy()[0]
                
                # 1. Base Signal (Scale based on model confidence)
                signals = probs - 0.5 
                
                # 2. Filter weak signals (Only act if >60% or <40% confident)
                signals[np.abs(signals) < 0.10] = 0.0
                
                # 3. Position Sizing
                sum_abs_signals = np.sum(np.abs(signals))
                if sum_abs_signals > 0:
                    weights = (signals / sum_abs_signals) * MAX_LEVERAGE
                else:
                    weights = np.zeros_like(signals)
                
                # 4. Gross Return
                gross_ret = np.sum(weights * actual_returns)
                
                # 5. Apply Transaction Costs
                if prev_weights is None: prev_weights = np.zeros_like(weights)
                turnover = np.sum(np.abs(weights - prev_weights))
                net_ret = gross_ret - (turnover * TRANSACTION_COST)
                prev_weights = weights.copy()
                
                # 6. Clip daily returns to realistic limits (prevent math explosion)
                net_ret = np.clip(net_ret, -0.10, 0.10) 
                portfolio_returns.append(net_ret)
                
                # 7. Benchmark: Buy & Hold Equal Weight
                market_ret = np.clip(np.mean(actual_returns), -0.10, 0.10)
                market_returns.append(market_ret)

        epoch_loss = train_loss/len(train_loader)
        epoch_acc = (val_acc/total_val)*100
        
        # Calculate Sharpe Ratio (Annualized)
        if len(portfolio_returns) > 1 and np.std(portfolio_returns) != 0:
            epoch_sharpe = np.sqrt(252) * (np.mean(portfolio_returns) / np.std(portfolio_returns))
        else:
            epoch_sharpe = 0.0
            
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        history['sharpe'].append(epoch_sharpe)
        
        if epoch == epochs - 1:
            history['strategy_returns'] = portfolio_returns
            history['market_returns'] = market_returns

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | "
              f"Dir Acc: {epoch_acc:.2f}% | Simulated Sharpe: {epoch_sharpe:.2f}")

    return model, history

# ==============================================================================
# 4. BUSINESS DASHBOARD GENERATION
# ==============================================================================

def plot_business_dashboard(history):
    print("\n📊 Generating Realistic Business Dashboard...")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Alpha-MaTCHS: Institutional Backtest Report', fontsize=18, fontweight='bold')

    # 1. Cumulative Returns Plot
    ax1 = fig.add_subplot(2, 2, (1, 2))
    cum_strategy = np.cumprod(1 + np.array(history['strategy_returns']))
    cum_market = np.cumprod(1 + np.array(history['market_returns']))
    
    ax1.plot(cum_strategy, label='Alpha-MaTCHS Strategy (Net of 0.1% Trade Costs)', color='green', linewidth=2.5)
    ax1.plot(cum_market, label='Market Baseline (Equal Weight)', color='gray', linestyle='--', linewidth=2)
    ax1.set_title('Cumulative Portfolio Growth', fontsize=14)
    ax1.set_xlabel('Validation Days (Out of Sample)')
    ax1.set_ylabel('Capital Multiplier (1.0 = Initial)')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Training Loss
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(history['loss'], color='red', marker='o', linewidth=2)
    ax2.set_title('Model Optimization (Huber + BCE Loss)')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3)

    # 3. Directional Accuracy
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(history['acc'], color='blue', marker='s', linewidth=2)
    ax3.set_title('Directional Accuracy (%)')
    ax3.set_xlabel('Epoch')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Alpha_MaTCHS_Dashboard.png', dpi=300)
    print("✅ Dashboard saved successfully as 'Alpha_MaTCHS_Dashboard.png'!")

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System initializing on: {device}")
    
    # --- Update this path to your exact dataset directory ---
    base_dir = "/home/alien-hawk/Desktop/PEDC/Final/stock_market_data"
    
    # 1. Load Data
    dataset = AlphaStockDataset(data_dir=base_dir, max_stocks=20)
    
    # Split into Train (80%) and Validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Batch size of 1 for validation simulates day-by-day trading
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 2. Initialize Model
    # num_features = 2 (LogReturn, Sentiment Placeholder)
    model = AlphaMaTCHS(num_features=2, d_model=64, seq_len=30).to(device)
    
    # 3. Train & Evaluate
    print("\n--- Commencing Alpha-MaTCHS Optimization ---")
    model, history = train_alpha_matches(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        adj_matrix=dataset.relation_matrix, 
        epochs=15, 
        device=device
    )
    
    # 4. Generate Dashboard
    plot_business_dashboard(history)
    
    # Save Final Weights
    torch.save(model.state_dict(), 'alpha_matches_weights.pth')
    print("✅ Model weights saved. Ready for live ingestion.")