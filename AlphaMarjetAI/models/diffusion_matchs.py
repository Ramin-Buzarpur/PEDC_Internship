import torch
import torch.nn as nn

from .relational_transformer import RelationalTransformer



class TemporalEncoder(nn.Module):

    def __init__(
        self,
        features,
        hidden
    ):
        super().__init__()

        self.net=nn.Sequential(

            nn.Conv1d(
                features,
                hidden,
                3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv1d(
                hidden,
                hidden,
                3,
                padding=1
            ),

            nn.GELU()

        )


    def forward(self,x):

        return self.net(x)



class AlphaMaTCHS(nn.Module):


    def __init__(
        self,
        features=10,
        hidden=128,
        seq_len=60
    ):

        super().__init__()


        self.encoder=TemporalEncoder(
            features,
            hidden
        )


        self.graph_encoder=RelationalTransformer(
            hidden*seq_len
        )


        self.regression=nn.Linear(
            hidden*seq_len,
            1
        )


        self.direction=nn.Linear(
            hidden*seq_len,
            1
        )


        self.mu=nn.Linear(
            hidden*seq_len,
            1
        )


        self.sigma=nn.Linear(
            hidden*seq_len,
            1
        )



    def forward(
        self,
        x,
        graph
    ):


        B,N,F,T=x.shape


        x=x.reshape(
            B*N,
            F,
            T
        )


        h=self.encoder(x)


        h=h.reshape(
            B,
            N,
            -1
        )


        h=self.graph_encoder(
            h,
            graph
        )


        reg=self.regression(h)


        direction=self.direction(h)


        mu=self.mu(h)

        sigma=torch.softplus(
            self.sigma(h)
        )


        return {
            "return":reg.squeeze(-1),
            "direction":direction.squeeze(-1),
            "mu":mu.squeeze(-1),
            "sigma":sigma.squeeze(-1)
        }