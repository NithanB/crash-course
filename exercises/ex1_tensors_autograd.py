"""
Exercise 1: Tensors & Autograd (~45 min)
Fill in every TODO, then run:  python3 ex1_tensors_autograd.py
Each part prints a check. Fix errors as they appear — that's the point.
"""
import torch
import torch.nn as nn
# ---------------------------------------------------------------
# Part A: Tensor basics
# ---------------------------------------------------------------
# 1. Create a 3x4 tensor of random values from a normal distribution.

a = torch.randn(3,4)

# 2. Create a 4x2 tensor of ones.
b = torch.tensor([[1.0, 1.0],
                  [1.0, 1.0],
                  [1.0, 1.0],       
                  [1.0, 1.0]])  # TODO

# 3. Matrix-multiply a and b -> shape (3, 2).
c = torch.mm(a,b)  # TODO

# 4. Compute the mean of each ROW of c -> shape (3,).
row_means = c.mean(dim=1)  # TODO

print("A:", c.shape == torch.Size([3, 2]), row_means.shape == torch.Size([3]))

# ---------------------------------------------------------------
# Part B: Reshaping (constant source of live-coding bugs)
# ---------------------------------------------------------------
# A fake batch of 8 grayscale 28x28 images:
imgs = torch.rand(8, 1, 28, 28)

# 5. Flatten to (8, 784) — as if feeding a Linear layer.
flat = nn.Flatten()(imgs)  # TODO

# 6. Add a batch dimension to a single image (1, 28, 28) -> (1, 1, 28, 28).
one = torch.rand(1, 28, 28)
batched = one.unsqueeze(0)  # TODO

print("B:", flat.shape == torch.Size([8, 784]), batched.shape == torch.Size([1, 1, 28, 28]))

# ---------------------------------------------------------------
# Part C: Autograd — gradient descent by hand
# Minimize f(w) = (w - 3)^2. The answer is obviously w = 3;
# make gradient descent find it.
# ---------------------------------------------------------------
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(100):
    # 7. Compute the loss f(w).
    loss = ((w-3)** 2).sum()  # TODO

    # 8. Backpropagate.
    loss.backward()  # backwardpropagation

    # 9. Update w with a gradient step. Two constraints:
    #    - wrap in torch.no_grad() (why? be ready to answer)
    #    - zero the gradient afterwards (why?)
    with torch.no_grad():
        w -= 0.1 * w.grad# TODO
    w.grad.zero_()  # TODO

print("C:", f"w = {w.item():.4f} (target 3.0)", abs(w.item() - 3.0) < 0.01)

# # ---------------------------------------------------------------
# # Verbal follow-ups to be able to answer:
# # - Why torch.no_grad() around the update?
# # - What happens if you never zero the gradient?
# # - What does requires_grad=True actually do?
# # ---------------------------------------------------------------
