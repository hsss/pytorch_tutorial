import os
import torch

x = torch.tensor(3, dtype = torch.float, requires_grad=True)
print(x)

y = 2. * x
print(y)

y.backward()
print(x.grad.item())
