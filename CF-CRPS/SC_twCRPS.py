
"""
SC-twCRPS v1.0 fixed

Fix:
SC_only failed because StudentT.sample() creates detached samples.
For gradient-based training we use rsample().

This tests gradient flow correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastNet(nn.Module):
    def __init__(self, d=20):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(d,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )

        self.mu = nn.Linear(32,1)
        self.sigma = nn.Linear(32,1)
        self.nu_raw = nn.Parameter(torch.tensor(1.5))

    def forward(self,x):
        h=self.body(x)

        mu=self.mu(h).squeeze(-1)
        sigma=F.softplus(
            self.sigma(h).squeeze(-1)
        )+1e-4

        nu=2.1+F.softplus(self.nu_raw)

        return mu,sigma,nu


def student_nll(y,mu,sigma,nu):

    z=(y-mu)/sigma

    logp=(
        torch.lgamma((nu+1)/2)
        -torch.lgamma(nu/2)
        -0.5*torch.log(nu*torch.pi)
        -torch.log(sigma)
        -((nu+1)/2)*torch.log1p(z*z/nu)
    )

    return -logp.mean()


def student_samples(mu,sigma,nu,n=100):

    dist=torch.distributions.StudentT(
        df=nu,
        loc=mu,
        scale=sigma
    )

    # critical fix: keep gradients
    return dist.rsample((n,)).transpose(0,1)


def mc_crps(samples,y):

    a=torch.abs(samples-y[:,None]).mean(1)

    h=samples.shape[1]//2

    b=torch.abs(
        samples[:,:h]-samples[:,h:2*h]
    ).mean(1)

    return (a-0.5*b).mean()


def sc_twcrps_penalty(mu,sigma,nu,y,vol,use_state=True):

    samples=student_samples(
        mu,sigma,nu
    )

    tail=torch.abs(y)>torch.quantile(
        torch.abs(y),0.9
    )

    if tail.sum()==0:
        return torch.tensor(0.,device=y.device,requires_grad=True)

    penalty=torch.abs(
        samples[tail]-y[tail,None]
    ).mean()

    if use_state:
        penalty=penalty*(1+vol.mean())

    return penalty


def generate_data(n=5000):

    x=torch.randn(n,20)
    regime=torch.rand(n)

    sigma=0.2+1.8*regime

    y=torch.randn(n)*sigma

    crash=torch.rand(n)<(0.03+0.07*regime)

    y[crash]-=torch.abs(
        torch.randn(crash.sum())*4
    )

    return x,y,regime


def evaluate(model,x,y):

    with torch.no_grad():

        mu,sigma,nu=model(x)

        samples=student_samples(
            mu,sigma,nu
        )

        return {
            "CRPS":mc_crps(samples,y).item(),
            "VaR95":(
                y>=torch.quantile(samples,.05,1)
            ).float().mean().item(),
            "VaR99":(
                y>=torch.quantile(samples,.01,1)
            ).float().mean().item()
        }


def run(method,seed):

    torch.manual_seed(seed)

    x,y,vol=generate_data()

    model=ForecastNet()

    opt=torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    grad_norm=0

    for epoch in range(60):

        mu,sigma,nu=model(x)

        nll=student_nll(y,mu,sigma,nu)

        sc=sc_twcrps_penalty(
            mu,sigma,nu,y,vol,
            use_state=("NoState" not in method)
        )

        if method=="StudentT":
            loss=nll
        elif method=="SC_only":
            loss=sc
        elif method=="Hybrid":
            loss=nll+0.05*sc
        else:
            loss=sc

        opt.zero_grad()
        loss.backward()

        if epoch==0:
            grad_norm=sum(
                p.grad.norm().item()
                for p in model.parameters()
                if p.grad is not None
            )

        opt.step()

    result=evaluate(model,x,y)
    result["GradNorm"]=grad_norm

    return result


if __name__=="__main__":

    for method in [
        "StudentT",
        "SC_only",
        "Hybrid",
        "SC_NoState"
    ]:

        results=[]

        for seed in [1,2,3,42,777]:
            results.append(
                run(method,seed)
            )

        print("\n",method)

        for k in results[0]:
            print(
                k,
                sum(r[k] for r in results)/len(results)
            )
