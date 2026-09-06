import torch, math_utils as m
torch.manual_seed(1)
z = torch.ones(1000, dtype=torch.float64)
a = (torch.rand(1000, dtype=torch.float64)*2+0.5)
b = (torch.rand(1000, dtype=torch.float64)*2+0.5)
err = (m.ibeta(z,a,b)-1).abs()
i = err.argmax()
print('max err float64:', err.max().item(),
      'at a=', a[i].item(), 'b=', b[i].item())
