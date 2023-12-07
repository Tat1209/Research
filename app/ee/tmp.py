import torch

t = torch.Tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

t = torch.stack([t for _ in range(3)])
t = t.view(t.shape[0], 4, -1)

print(t)
print(t.shape)