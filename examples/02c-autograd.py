import os
import torch

x = torch.tensor(3, dtype = torch.float, requires_grad=True)
print(x)

y = x ** 2.
print(y)

z = 2 * y
z.backward()
print(x.grad.item())
