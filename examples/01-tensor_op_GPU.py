import os
import torch

print(torch.cuda.is_available())

A = torch.tensor([1, 2, 3], dtype = torch.float).to('cuda')
B = torch.tensor([4, 5, 6], dtype = torch.float).to('cuda')

result = torch.dot(A, B)

print(result)
print(result.item())


