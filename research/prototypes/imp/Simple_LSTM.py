import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

ticker = "^GSPC"
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# (Close)
df = data[['Close']].copy()
df.columns = ['Close']

# bazdeh
df['Return'] = df['Close'].pct_change()

df.dropna(inplace=True)

print("first data:")
print(df.head())

def calculate_ewma_volatility(returns, span=60):
    ewma_variance = returns.ewm(span=span).var()
    ewma_vol = np.sqrt(ewma_variance)
    return ewma_vol

df['Volatility'] = calculate_ewma_volatility(df['Return'])
df.dropna(inplace=True)

df['Norm_Return'] = df['Return'] / df['Volatility']
#12 days and 26 days
ewma_12 = df['Close'].ewm(span=12, adjust=False).mean()
ewma_26 = df['Close'].ewm(span=26, adjust=False).mean()

df['MACD_Raw'] = ewma_12 - ewma_26

rolling_std_63 = df['Close'].rolling(window=63).std()
df['MACD_Norm'] = df['MACD_Raw'] / rolling_std_63

df.dropna(inplace=True)

print("\n Date Preprocessed:")
print(df[['Return', 'Volatility', 'Norm_Return', 'MACD_Norm']].tail())

class SharpeLoss(nn.Module):
    def __init__(self, target_vol=0.10, eps=1e-6):

        super(SharpeLoss, self).__init__()
        self.target_vol = target_vol
        self.eps = eps
        self.annualization_factor = np.sqrt(252)

    def forward(self, predictions, true_returns, current_volatilities):
        vs_factor = self.target_vol / (current_volatilities + self.eps)
        weights = predictions * vs_factor
        strategy_returns = weights * true_returns
        mean_return = torch.mean(strategy_returns)
        var_return = torch.var(strategy_returns)
        
        sharpe_ratio = (mean_return / torch.sqrt(var_return + self.eps)) * self.annualization_factor
        return -sharpe_ratio

class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim):

        super(ProjectionHead, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, h_t):
        signal = self.tanh(self.linear(h_t))
        return signal
    
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):

        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        self.projection = ProjectionHead(hidden_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last_hidden_state = out[:, -1, :]
        signal = self.projection(last_hidden_state)
        
        return signal

dummy_input = torch.randn(16, 84, 3)
model = SimpleLSTM(input_dim=3, hidden_dim=64)
dummy_output = model(dummy_input)

print("\n--- Model Test ---")
print("Input shape:", dummy_input.shape)
print("Output shape:", dummy_output.shape)
print("Sample output signals:\n", dummy_output[:5].detach().numpy())

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

def create_sliding_windows(df, lookback=84):

    X_features = []
    y_returns = []
    y_vols = []
    
    feature_cols = ['Norm_Return', 'MACD_Norm']
    
    data_features = df[feature_cols].values
    data_returns = df['Return'].values
    data_vols = df['Volatility'].values
    
    for i in range(len(df) - lookback):
        X_features.append(data_features[i : i+lookback])
        
        y_returns.append(data_returns[i+lookback])
        y_vols.append(data_vols[i+lookback])
        
    X_tensor = torch.tensor(np.array(X_features), dtype=torch.float32)
    y_ret_tensor = torch.tensor(np.array(y_returns), dtype=torch.float32)
    y_vol_tensor = torch.tensor(np.array(y_vols), dtype=torch.float32)
    
    return X_tensor, y_ret_tensor, y_vol_tensor

lookback_window = 84
X, y_ret, y_vol = create_sliding_windows(df, lookback=lookback_window)

print(f"\n shape of tensor's input: {X.shape}") 

batch_size = 128
epochs = 20
learning_rate = 0.001

dataset = TensorDataset(X, y_ret, y_vol)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SimpleLSTM(input_dim=2, hidden_dim=64, num_layers=1)
criterion = SharpeLoss(target_vol=0.10)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("\n--- Train Started---")
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    
    for batch_X, batch_ret, batch_vol in dataloader:
        optimizer.zero_grad()
        
        predictions = model(batch_X).squeeze() 
        
        loss = criterion(predictions, batch_ret, batch_vol)
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    
    print(f"Epoch [{epoch+1}/{epochs}] | Sharpe Ratio Score: {-avg_loss:.4f}")

print("Train Finished")

print("\n--- Model Evaluation & Portfolio Construction ---")
model.eval()
with torch.no_grad():
    predictions = model(X).squeeze().numpy()

eval_df = df.iloc[lookback_window:].copy()
eval_df['Model_Signal'] = predictions

target_vol = 0.40
eval_df['vs_factor'] = 1.0 / eval_df['Volatility']
eval_df['Weight'] = eval_df['Model_Signal'] * target_vol * eval_df['vs_factor']

eval_df['Strategy_Return'] = eval_df['Weight'].shift(1) * eval_df['Return']

eval_df.dropna(inplace=True)

eval_df['Cumulative_Market'] = (1 + eval_df['Return']).cumprod()
eval_df['Cumulative_Strategy'] = (1 + eval_df['Strategy_Return']).cumprod()

mean_strat_ret = eval_df['Strategy_Return'].mean()
std_strat_ret = eval_df['Strategy_Return'].std()
annualized_sharpe = (mean_strat_ret / std_strat_ret) * np.sqrt(252)

print(f"Annualized Strategy Sharpe Ratio: {annualized_sharpe:.4f}")
print(f"Cumulative Return (Strategy): {eval_df['Cumulative_Strategy'].iloc[-1]:.4f}")
print(f"Cumulative Return (Market): {eval_df['Cumulative_Market'].iloc[-1]:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(eval_df.index, eval_df['Cumulative_Strategy'], label='LSTM Strategy (Vol Targeted)', color='blue')
plt.plot(eval_df.index, eval_df['Cumulative_Market'], label='Market Baseline (S&P 500)', color='gray', alpha=0.7)

plt.title("Cumulative PnL: LSTM Strategy vs Market Baseline")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()