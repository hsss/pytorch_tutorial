import os
import torch

def reset_grad(p):
	if p.grad is not None:
		p.grad.detach_()
		p.grad.zero_()

x = torch.tensor(list(range(10)), dtype=torch.float)
y = 2. * x + 0.5
print(x)
print(y)

W = torch.tensor(0.1, dtype=torch.float, requires_grad = True)
b = torch.tensor(0., dtype=torch.float, requires_grad = True)
print(W)
print(b)


for i  in range(1000):
	reset_grad(W)
	reset_grad(b)

	output = W * x + b
	loss = ((output - y) ** 2.).mean() ** 0.5

	print(loss.item())
	
	loss.backward()
		
	W.data.add_(-0.001, W.grad.data)
	b.data.add_(-0.001, b.grad.data)

print(W)
print(b)















