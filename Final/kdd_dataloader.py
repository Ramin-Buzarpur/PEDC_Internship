import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultiMarketStockDataset(Dataset):
    def __init__(self, data_dir, markets=["nasdaq", "nyse", "sp500", "forbes2000"], 
                 seq_len_past=30, seq_len_future=10, max_stocks=50, corr_threshold=0.6):
        self.seq_len_past = seq_len_past
        self.seq_len_future = seq_len_future
        self.total_seq_len = seq_len_past + seq_len_future
        
        all_csv_files = []
        # اسکن کردن تمام پوشه‌های بازار
        for market in markets:
            csv_path = os.path.join(data_dir, market, "csv")
            if os.path.exists(csv_path):
                files = glob.glob(os.path.join(csv_path, "*.csv")) + glob.glob(os.path.join(csv_path, "*.CSV"))
                all_csv_files.extend(files)
        
        # حذف تکراری‌ها و محدود کردن تعداد کل سهام برای جلوگیری از کمبود رم
        all_csv_files = sorted(list(set(all_csv_files)))[:max_stocks]
        self.num_stocks = len(all_csv_files)
        
        if self.num_stocks == 0:
            raise ValueError(f"No CSV files found in {data_dir}. Check directory paths!")
            
        print(f"Successfully loaded {self.num_stocks} stocks across all markets.")
        
        dataframes = {}
        close_prices = {}
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for file in all_csv_files:
            stock_name = os.path.basename(file).replace(".csv", "").replace(".CSV", "")
            try:
                df = pd.read_csv(file)
                # پیدا کردن ستون تاریخ پویا
                date_col = next((col for col in ['Date', 'date', 'Timestamp', 'time'] if col in df.columns), None)
                if date_col is None:
                    continue
                
                df['Date'] = pd.to_datetime(df[date_col])
                df.set_index('Date', inplace=True)
                
                available_features = [f for f in features if f in df.columns]
                if len(available_features) < 4:
                    continue
                    
                dataframes[stock_name] = df[available_features]
                if 'Close' in df.columns:
                    close_prices[stock_name] = df['Close']
            except Exception as e:
                pass
        
        if len(dataframes) == 0:
            raise ValueError("No valid stock dataframes could be parsed.")
            
        self.num_stocks = len(dataframes)
        
        # همگام‌سازی تاریخ‌ها بین تمام سهام‌ها
        merged_df = pd.concat(dataframes, axis=1, join='inner')
        merged_df.ffill(inplace=True)
        merged_df.bfill(inplace=True)
        
        self.raw_data = merged_df.values
        self.num_features = len(features)
        
        # نرمال‌سازی Z-Score
        self.mean = np.mean(self.raw_data, axis=0)
        self.std = np.std(self.raw_data, axis=0) + 1e-8
        self.norm_data = (self.raw_data - self.mean) / self.std
        self.norm_data = self.norm_data.reshape(-1, self.num_stocks, self.num_features)
        
        # ساخت ماتریس روابط (گراف همبستگی)
        print("Building Global Correlation Graph Matrix...")
        close_df = pd.DataFrame(close_prices).dropna(how='all')
        close_df.ffill(inplace=True)
        close_df.bfill(inplace=True)
        
        returns_df = close_df.pct_change().dropna()
        corr_matrix = returns_df.corr().values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # ساخت ماتریس روابط 3 بُعدی [N, N, 2] (همبستگی مثبت و منفی)
        self.relation_matrix = np.zeros((self.num_stocks, self.num_stocks, 2), dtype=np.float32)
        self.relation_matrix[:, :, 0] = (corr_matrix > corr_threshold).astype(np.float32)
        self.relation_matrix[:, :, 1] = (corr_matrix < -corr_threshold).astype(np.float32)
        self.relation_matrix = torch.tensor(self.relation_matrix, dtype=torch.float32)
        
        self.valid_indices = len(self.norm_data) - self.total_seq_len

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        window = self.norm_data[idx : idx + self.total_seq_len]
        past_cond = window[:self.seq_len_past]      # (L_past, N, P)
        future_target = window[self.seq_len_past:]  # (L_future, N, P)
        
        past_cond = torch.tensor(past_cond, dtype=torch.float32).permute(1, 2, 0)
        future_target = torch.tensor(future_target, dtype=torch.float32).permute(1, 2, 0)
        
        return past_cond, future_target, self.relation_matrix

# ==========================================
# نحوه تست و اجرا
# ==========================================
if __name__ == "__main__":
    base_dir = "/home/alien-hawk/Desktop/PEDC/Final/stock_market_data"
    
    # بارگذاری ترکیبی از تمام بازارها با انتخاب مثلاً ۳۰ سهم برتر
    dataset = MultiMarketStockDataset(
        data_dir=base_dir, 
        markets=["nasdaq", "nyse", "sp500", "forbes2000"], 
        max_stocks=30, 
        corr_threshold=0.5
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for past, future, rel_matrix in dataloader:
        print(f"\n--- Output Check ---")
        print(f"Past Condition Shape: {past.shape}")         # [Batch, N, Features, 30]
        print(f"Future Target Shape : {future.shape}")       # [Batch, N, Features, 10]
        print(f"Relation Matrix Shape: {rel_matrix.shape}")    # [N, N, 2]
        break