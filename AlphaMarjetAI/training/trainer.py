# losses/safps_crps.py

import torch



def crps_loss(
    samples,
    target
):


    """
    Continuous Ranked Probability Score

    samples:
        [batch, scenarios]

    target:
        [batch]
    """

    term1=torch.mean(
        torch.abs(
            samples-target.unsqueeze(1)
        )
    )


    term2=0.5*torch.mean(
        torch.abs(
            samples[:,None,:]
            -
            samples[:,:,None]
        )
    )


    return term1-term2




def tail_weighted_crps(
    samples,
    target,
    risk_weight=2.0
):


    error=torch.abs(
        samples-target.unsqueeze(1)
    )


    weights=torch.ones_like(
        error
    )


    weights[
        samples < target.unsqueeze(1)
    ] *= risk_weight


    return torch.mean(
        error*weights
    )





class SAFPSLoss(torch.nn.Module):


    def __init__(
        self,
        tail_weight=2.0
    ):

        super().__init__()

        self.tail_weight=tail_weight



    def forward(
        self,
        samples,
        target
    ):


        normal=crps_loss(
            samples,
            target
        )


        tail=tail_weighted_crps(
            samples,
            target,
            self.tail_weight
        )


        return (
            0.5*normal+
            0.5*tail
        )