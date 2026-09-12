# datasets/unified_market_dataset.py

import os
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class UnifiedMarketDataset(Dataset):

    def __init__(
        self,
        data_dir,
        markets=[
            "nasdaq",
            "nyse",
            "sp500",
            "forbes2000"
        ],
        seq_len=60,
        future_len=10,
        max_stocks=200,
        corr_threshold=0.5
    ):

        self.seq_len = seq_len
        self.future_len = future_len


        ###################################
        # 1. Collect All Markets
        ###################################

        files=[]

        for market in markets:

            path=os.path.join(
                data_dir,
                market,
                "csv"
            )

            if os.path.exists(path):

                files += glob.glob(
                    path+"/*.csv"
                )


        files=sorted(
            list(set(files))
        )


        if max_stocks:

            files=files[:max_stocks]


        print(
            f"Loading {len(files)} stocks"
        )


        ###################################
        # 2. Load OHLCV
        ###################################

        stocks={}


        for file in files:

            name=os.path.basename(
                file
            ).replace(".csv","")


            try:

                df=pd.read_csv(
                    file
                )


                date_col=None

                for c in [
                    "Date",
                    "date",
                    "Timestamp"
                ]:

                    if c in df.columns:
                        date_col=c


                if date_col is None:
                    continue


                df["Date"]=pd.to_datetime(
                    df[date_col]
                )


                df=df.set_index(
                    "Date"
                )


                cols=[
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]


                if not all(
                    c in df.columns
                    for c in cols
                ):
                    continue


                stocks[name]=df[cols]


            except:

                continue



        ###################################
        # 3. Merge Markets
        ###################################


        merged=pd.concat(
            stocks,
            axis=1,
            join="inner"
        )


        merged.ffill(
            inplace=True
        )

        merged.dropna(
            inplace=True
        )



        self.symbols=list(
            stocks.keys()
        )


        ###################################
        # 4. Feature Engineering
        ###################################


        features=[]


        for stock in self.symbols:


            df=merged[stock]


            close=df["Close"]


            log_return=np.log(
                close /
                close.shift(1)
            )


            volatility=(
                log_return
                .rolling(5)
                .std()
            )


            momentum=(
                close /
                close.shift(10)
                -1
            )


            volume_change=(
                df["Volume"]
                /
                df["Volume"]
                .shift(5)
                -1
            )


            price_range=(
                df["High"]
                -
                df["Low"]
            ) / close



            f=pd.DataFrame({

                "open":
                df["Open"],

                "high":
                df["High"],

                "low":
                df["Low"],

                "close":
                df["Close"],

                "volume":
                df["Volume"],

                "return":
                log_return,

                "volatility":
                volatility,

                "momentum":
                momentum,

                "volume_change":
                volume_change,

                "range":
                price_range

            })


            features.append(
                f
            )



        feature_df=pd.concat(
            features,
            axis=1
        )


        feature_df.fillna(
            0,
            inplace=True
        )



        ###################################
        # 5. Normalize
        ###################################

        data=feature_df.values


        mean=data.mean(
            axis=0
        )

        std=data.std(
            axis=0
        )+1e-8


        data=(data-mean)/std



        self.num_features=10


        self.data=data.reshape(
            len(data),
            len(self.symbols),
            self.num_features
        )



        ###################################
        # 6. Graph
        ###################################


        returns=[]


        for stock in self.symbols:

            returns.append(

                merged[stock]["Close"]
                .pct_change()
                .fillna(0)
                .values

            )


        returns=np.array(
            returns
        ).T



        corr=np.corrcoef(
            returns.T
        )


        graph=np.zeros(
            (
            len(self.symbols),
            len(self.symbols),
            2
            )
        )



        graph[:,:,0]=(
            corr >
            corr_threshold
        )


        graph[:,:,1]=(
            corr <
            -corr_threshold
        )



        np.fill_diagonal(
            graph[:,:,0],
            1
        )



        self.graph=torch.tensor(
            graph,
            dtype=torch.float32
        )



        self.length=(
            len(self.data)
            -
            seq_len
            -
            future_len
        )



        print(
            "Dataset ready:",
            self.length
        )



    def __len__(self):

        return self.length



    def __getitem__(self,index):


        past=self.data[
            index:
            index+self.seq_len
        ]


        future=self.data[
            index+self.seq_len:
            index+self.seq_len+self.future_len
        ]


        past=torch.tensor(
            past,
            dtype=torch.float32
        ).permute(
            1,2,0
        )


        future=torch.tensor(
            future,
            dtype=torch.float32
        ).permute(
            1,2,0
        )


        return (
            past,
            future,
            self.graph
        )