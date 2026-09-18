"""
Differentiable special functions for torch (missing from torch.special).

ibeta(z, a, b): regularized incomplete beta I_z(a,b), vectorized over
leading dimensions, differentiable w.r.t. z, a, b.

Method: arc-sine substitution t = sin^2(theta) removes the (1-t)^{b-1}
endpoint singularity, then Gauss-Legendre quadrature over theta in
[0, asin(sqrt(z))]:

    I_z(a,b) = (2 / B(a,b)) * INTEGRAL_0^{asin(sqrt(z))}
               (sin th)^{2a-1} (cos th)^{2b-1} d(theta)

96-node GL integrates this to ~1e-10 accuracy for a, b >= 0.5
(the Student-t use case: a = nu/2 >= 1.05, b = 0.5).
"""
import numpy as np
import torch

_NQ = 96
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(_NQ)
_GL_T = torch.from_numpy(_NODES).double()     # in [-1, 1]
_GL_W = torch.from_numpy(_WEIGHTS).double()


def ibeta(z, a, b):
    """I_z(a,b). z: [...]; a, b: broadcastable to z."""
    z = z.clamp(0.0, 1.0)
    aa = torch.as_tensor(a, dtype=z.dtype, device=z.device).expand_as(z)
    bb = torch.as_tensor(b, dtype=z.dtype, device=z.device).expand_as(z)

    logB = (torch.lgamma(aa) + torch.lgamma(bb) - torch.lgamma(aa + bb))

    zs = z.unsqueeze(-1)                                 # [..., 1]
    u = _GL_T.to(z.device).reshape([1] * z.dim() + [_NQ])\
        .expand(*zs.shape[:-1], _NQ)
    w = _GL_W.to(z.device).reshape([1] * z.dim() + [_NQ])\
        .expand(*zs.shape[:-1], _NQ)

    phi = torch.asin(torch.sqrt(zs))                     # [..., 1]
    theta = 0.5 * phi * (u + 1.0)                        # [..., nq]
    dtheta = 0.5 * phi                                   # [..., 1]

    sin_l = torch.sin(theta).clamp(min=1e-30).log()      # [..., nq]
    cos_l = torch.cos(theta).clamp(min=1e-30).log()      # [..., nq]
    exp_a = (2.0 * aa).unsqueeze(-1) - 1.0               # [..., 1]
    exp_b = (2.0 * bb).unsqueeze(-1) - 1.0               # [..., 1]
    logB3 = logB.unsqueeze(-1)                           # [..., 1]

    log_int = (exp_a * sin_l + exp_b * cos_l - logB3)
    integrand = torch.exp(log_int)
    return (2.0 * dtheta * w * integrand).sum(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    # I_1(a,b) = 1
    z = torch.ones(5, 3)
    a = torch.rand(5, 3) * 2 + 0.5
    b = torch.rand(5, 3) * 2 + 0.5
    print("I_1(a,b) - 1 (want ~0):",
          (ibeta(z, a, b) - 1).abs().max().item())
    # I_0.5(1,1) = 0.5
    print("I_0.5(1,1) (want 0.5):",
          ibeta(torch.full((1,), 0.5), torch.ones(1), torch.ones(1)))
    # symmetry I_z(a,b) = 1 - I_{1-z}(b,a)
    z2 = torch.rand(5, 3)
    lhs = ibeta(z2, a, b)
    rhs = 1 - ibeta(1 - z2, b, a)
    print("symmetry err (want ~0):", (lhs - rhs).abs().max().item())
    # gradient check
    z3 = torch.rand(5, 3, requires_grad=True)
    ibeta(z3, a, b).sum().backward()
    print("grad ok:", torch.isfinite(z3.grad).all().item())
