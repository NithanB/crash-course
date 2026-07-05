"""
Exercise 2: MLP on MNIST — THE core exercise (~90 min first pass)
This is the most likely shape of the live exercise. By end of day you
should be able to write this from a BLANK file without notes.

Fill in every TODO, then run:  python3 ex2_mlp_mnist.py
Target: >95% test accuracy after 2 epochs (runs in ~1-2 min on CPU).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------
# 1. Data: MNIST train and test sets, with ToTensor transform.
#    Wrap each in a DataLoader (batch_size=64, shuffle train only).
# ---------------------------------------------------------------
transform = transforms.ToTensor()

train_ds = ...  # TODO: datasets.MNIST(...)
test_ds = ...   # TODO

train_loader = ...  # TODO
test_loader = ...   # TODO

# ---------------------------------------------------------------
# 2. Model: an MLP as an nn.Module subclass.
#    784 -> 128 -> ReLU -> 10 (logits, no softmax — know why!)
# ---------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: define layers (tip: nn.Flatten() first)

    def forward(self, x):
        ...  # TODO

model = MLP()

# ---------------------------------------------------------------
# 3. Loss and optimizer: cross-entropy + Adam (lr=1e-3).
# ---------------------------------------------------------------
criterion = ...  # TODO
optimizer = ...  # TODO

# ---------------------------------------------------------------
# 4. Training loop: 2 epochs. The 5 sacred steps per batch:
#    zero_grad -> forward -> loss -> backward -> step
#    Print average loss per epoch.
# ---------------------------------------------------------------
for epoch in range(2):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        ...  # TODO

    print(f"epoch {epoch + 1}: avg loss {total_loss / len(train_loader):.4f}")

# ---------------------------------------------------------------
# 5. Evaluation: accuracy on the test set.
#    Remember: model.eval() and torch.no_grad().
#    Prediction = argmax over the 10 logits.
# ---------------------------------------------------------------
model.eval()
correct, total = 0, 0
# TODO: loop over test_loader, count correct predictions

print(f"test accuracy: {correct / total:.4f}")  # want > 0.95

# ---------------------------------------------------------------
# Verbal follow-ups to be able to answer:
# - Why no softmax in the model?
# - Why shuffle the training data but not the test data?
# - What would you change to reduce overfitting? (add nn.Dropout)
# - What does model.eval() change?
# ---------------------------------------------------------------
