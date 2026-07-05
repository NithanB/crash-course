"""
Exercise 1: Tensors & Autograd (~45 min)
Fill in every TODO, then run:  python3 ex1_tensors_autograd.py
Each part prints a check. Fix errors as they appear — that's the point.
"""
import torch

# ---------------------------------------------------------------
# Part A: Tensor basics
# ---------------------------------------------------------------
# 1. Create a 3x4 tensor of random values from a normal distribution.
a = ...  # TODO

# 2. Create a 4x2 tensor of ones.
b = ...  # TODO

# 3. Matrix-multiply a and b -> shape (3, 2).
c = ...  # TODO

# 4. Compute the mean of each ROW of c -> shape (3,).
row_means = ...  # TODO

print("A:", c.shape == torch.Size([3, 2]), row_means.shape == torch.Size([3]))

# ---------------------------------------------------------------
# Part B: Reshaping (constant source of live-coding bugs)
# ---------------------------------------------------------------
# A fake batch of 8 grayscale 28x28 images:
imgs = torch.rand(8, 1, 28, 28)

# 5. Flatten to (8, 784) — as if feeding a Linear layer.
flat = ...  # TODO

# 6. Add a batch dimension to a single image (1, 28, 28) -> (1, 1, 28, 28).
one = torch.rand(1, 28, 28)
batched = ...  # TODO

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
    loss = ...  # TODO

    # 8. Backpropagate.
    ...  # TODO

    # 9. Update w with a gradient step. Two constraints:
    #    - wrap in torch.no_grad() (why? be ready to answer)
    #    - zero the gradient afterwards (why?)
    ...  # TODO

print("C:", f"w = {w.item():.4f} (target 3.0)", abs(w.item() - 3.0) < 0.01)

# ---------------------------------------------------------------
# Verbal follow-ups to be able to answer:
# - Why torch.no_grad() around the update?
# - What happens if you never zero the gradient?
# - What does requires_grad=True actually do?
# ---------------------------------------------------------------
