"""Solution — Exercise 1: Tensors & Autograd"""
import torch

# Part A
a = torch.randn(3, 4)
b = torch.ones(4, 2)
c = a @ b                      # or torch.matmul(a, b)
row_means = c.mean(dim=1)      # dim=1 collapses columns -> one mean per row

print("A:", c.shape == torch.Size([3, 2]), row_means.shape == torch.Size([3]))

# Part B
imgs = torch.rand(8, 1, 28, 28)
flat = imgs.view(8, -1)        # or imgs.flatten(start_dim=1)
one = torch.rand(1, 28, 28)
batched = one.unsqueeze(0)

print("B:", flat.shape == torch.Size([8, 784]), batched.shape == torch.Size([1, 1, 28, 28]))

# Part C
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(100):
    loss = (w - 3) ** 2
    loss.backward()
    with torch.no_grad():      # update must not be tracked by autograd
        w -= lr * w.grad
    w.grad.zero_()             # otherwise gradients ACCUMULATE

print("C:", f"w = {w.item():.4f} (target 3.0)", abs(w.item() - 3.0) < 0.01)

# Answers to verbal follow-ups:
# - no_grad(): the weight update is not part of the loss computation; without it
#   autograd would track the update op into the graph (and error on a leaf tensor).
# - Never zeroing: .grad sums every backward() call -> updates use stale gradients.
# - requires_grad=True: tells autograd to build a graph of operations on this
#   tensor so backward() can compute dloss/dw into w.grad.
