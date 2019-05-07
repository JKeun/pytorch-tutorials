import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

out.backward(torch.tensor(1.))

print(x.grad)




x1 = torch.randn(3, requires_grad=True)
y1 = x1 * 2

v = torch.tensor([1, 2, 3], dtype=torch.float)
y1.backward(v)

print(x1.grad)
