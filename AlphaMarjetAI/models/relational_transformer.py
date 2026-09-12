import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelationalTransformer(nn.Module):

    def __init__(
        self,
        embed_dim,
        heads=4
    ):
        super().__init__()

        self.heads=heads
        self.qkv=nn.Linear(
            embed_dim,
            embed_dim*3
        )

        self.proj=nn.Linear(
            embed_dim,
            embed_dim
        )

        self.norm=nn.LayerNorm(
            embed_dim
        )


    def forward(
        self,
        x,
        graph
    ):

        B,N,D=x.shape

        H=self.heads

        qkv=self.qkv(x)

        q,k,v=torch.chunk(
            qkv,
            3,
            dim=-1
        )


        score=torch.matmul(
            q,
            k.transpose(-2,-1)
        ) / math.sqrt(D)


        if graph is not None:

            mask=graph[:,:,0]

            score=score.masked_fill(
                mask.unsqueeze(0)==0,
                -1e9
            )


        attn=F.softmax(
            score,
            dim=-1
        )


        out=torch.matmul(
            attn,
            v
        )


        return self.norm(
            x+self.proj(out)
        )