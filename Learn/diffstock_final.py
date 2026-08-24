"""A corrected, self-contained DiffStock/MaTCHS-style forecaster.

Example:
  python Learn/diffstock_final.py --data Learn/nasdq.csv --epochs 30 --samples 50

The CSV supplied with this project contains one stock and exogenous indicators,
therefore N=1 is used (relations are still implemented and become useful when
multi-stock data are supplied).
"""
from __future__ import annotations
import argparse, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class NoiseSchedule:
    def __init__(self, steps, device):
        self.steps = steps
        self.betas = torch.linspace(1e-4, .2, steps, device=device)
        self.alphas = 1 - self.betas
        self.ab = torch.cumprod(self.alphas, 0)
        self.ab_prev = torch.cat([torch.ones(1, device=device), self.ab[:-1]])
        self.posterior_var = self.betas * (1-self.ab_prev) / (1-self.ab)

    def q(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.ab[t].view(-1, 1, 1, 1)
        return a.sqrt()*x0 + (1-a).sqrt()*noise, noise

class MaTCHS(nn.Module):
    def __init__(self, features, cond_len, horizon, width=64, heads=4, layers=2):
        super().__init__(); self.horizon=horizon
        self.temporal = nn.Sequential(nn.Conv1d(features, width, 3, padding=1),
                                      nn.GELU(), nn.Conv1d(width, width, 3, padding=1))
        self.noisy = nn.Conv1d(1, width, 3, padding=1)
        self.t_embed = nn.Sequential(nn.Linear(1,width), nn.SiLU(), nn.Linear(width,width))
        flat = width * (cond_len + horizon)
        enc = nn.TransformerEncoderLayer(flat, heads, 4*flat, batch_first=True,
                                         norm_first=True, dropout=.1)
        self.spatial = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.Conv1d(width, width, 3, padding=1),
                                 nn.GELU(), nn.Conv1d(width, 1, 1))
    def forward(self, noisy, t, cond, relation):
        # cond [B,N,P,L], noisy [B,N,1,H]
        b,n,p,l=cond.shape; h=noisy.shape[-1]
        hc=self.temporal(cond.reshape(b*n,p,l))
        hz=self.noisy(noisy.reshape(b*n,1,h))
        z=torch.cat([hc, hz], -1)
        te=self.t_embed(t[:,None].float()/1000).repeat_interleave(n,0)[:,:,None]
        z=z+te
        # Flatten each stock's temporal representation; attention is across stocks.
        z=z.reshape(b,n,-1)
        mask=(relation==0).float().masked_fill(relation==0, -1e4)
        z=self.spatial(z, mask=mask)
        z=z.reshape(b*n, z.shape[-1]//(l+h), l+h)
        return self.out(z)[:,:,-h:].reshape(b,n,1,h)

def windows(arr, cond, horizon, split, close_scale):
    tr=arr[:split]; te=arr[split-cond:]
    def make(x):
        xs=[]; ys=[]
        for i in range(len(x)-cond-horizon+1):
            xs.append(x[i:i+cond].T)
            last=x[i+cond-1,3]
            ys.append(((x[i+cond:i+cond+horizon,3]-last)/close_scale)[None])
        return torch.tensor(np.asarray(xs),dtype=torch.float32)[:,None], torch.tensor(np.asarray(ys),dtype=torch.float32)[:,None]
    return make(tr), make(te)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="Learn/nasdq.csv")
    ap.add_argument("--cond",type=int,default=30); ap.add_argument("--horizon",type=int,default=10)
    ap.add_argument("--steps",type=int,default=100); ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--batch",type=int,default=32); ap.add_argument("--width",type=int,default=64)
    ap.add_argument("--samples",type=int,default=20); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--no-plot",action="store_true"); args=ap.parse_args(); seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df=pd.read_csv(args.data).sort_values("Date")
    cols=["Open","High","Low","Close","Volume","InterestRate","ExchangeRate","VIX","TEDSpread","EFFR","Gold","Oil"]
    x=df[cols].replace([np.inf,-np.inf],np.nan).ffill().bfill().to_numpy(np.float32)
    split=int(.8*len(x)); mu=x[:split].mean(0); sd=x[:split].std(0); sd[sd<1e-6]=1
    xn=(x-mu)/sd; (trc,try_), (tec,tey)=windows(xn,args.cond,args.horizon,split,1.0)
    model=MaTCHS(len(cols),args.cond,args.horizon,args.width).to(device); diff=NoiseSchedule(args.steps,device)
    rel=torch.ones(1,1,device=device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(trc,try_),args.batch,shuffle=True)
    model.train()
    for ep in range(args.epochs):
        total=0
        for cond,y in loader:
            cond,y=cond.to(device),y.to(device); t=torch.randint(0,args.steps,(len(y),),device=device)
            noisy,noise=diff.q(y,t); pred=model(noisy,t,cond,rel)
            loss=(pred-noise).pow(2).mean(); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(y)
        if (ep+1)%10==0: print(f"epoch {ep+1:03d} loss={total/len(trc):.5f}")
    model.eval(); cond,y=tec[-1:].to(device),tey[-1:].numpy()[0,0,0]
    preds=[]
    with torch.no_grad():
        for _ in range(args.samples):
            z=torch.randn_like(torch.zeros((1,1,1,args.horizon),device=device))
            for i in range(args.steps-1,-1,-1):
                t=torch.full((1,),i,device=device,dtype=torch.long); eps=model(z,t,cond,rel)
                a=diff.alphas[i]; ab=diff.ab[i]; mean=(z-(1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                z=mean+(diff.posterior_var[i].clamp_min(1e-20).sqrt()*torch.randn_like(z) if i else 0)
            preds.append(z.cpu().numpy()[0,0,0])
    anchor=cond.cpu().numpy()[0,0,3,-1]*sd[3]+mu[3]
    pred=anchor+np.mean(preds,0)*sd[3]; truth=anchor+y*sd[3]
    mae=np.mean(np.abs(pred-truth)); rmse=np.sqrt(np.mean((pred-truth)**2))
    print(f"test MAE={mae:.4f} RMSE={rmse:.4f}")
    if not args.no_plot:
        import matplotlib.pyplot as plt
        past=cond.cpu().numpy()[0,0,3]*sd[3]+mu[3]
        plt.plot(range(args.cond),past,label="history"); plt.plot(range(args.cond,args.cond+args.horizon),truth,label="actual")
        plt.plot(range(args.cond,args.cond+args.horizon),pred,"--",label="prediction"); plt.legend(); plt.tight_layout(); plt.show()
if __name__=="__main__": main()
