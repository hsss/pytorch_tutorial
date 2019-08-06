import os
import torch

A = torch.tensor([1, 2, 3], dtype = torch.float)
B = torch.tensor([4, 5, 6], dtype = torch.float)

result = torch.dot(A, B)

print(result)
print(result.item())


