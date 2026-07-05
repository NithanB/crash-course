"""Solution — Exercise 2: MLP on MNIST
Run: python3 ex2_solution.py          (real MNIST, ~1-2 min CPU, >95% acc)
     python3 ex2_solution.py --fake   (tiny synthetic data, just a smoke test)
"""
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

FAKE = "--fake" in sys.argv
transform = transforms.ToTensor()

# 1. Data
if FAKE:  # offline smoke test only — accuracy will be ~random
    fake = lambda n: datasets.FakeData(size=n, image_size=(1, 28, 28),
                                       num_classes=10, transform=transform)
    train_ds, test_ds = fake(512), fake(256)
else:
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)   # no shuffle: order irrelevant at eval

# 2. Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),            # (B,1,28,28) -> (B,784)
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),      # logits; CrossEntropyLoss applies log-softmax
        )

    def forward(self, x):
        return self.net(x)

model = MLP()

# 3. Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Training loop
for epoch in range(2):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch + 1}: avg loss {total_loss / len(train_loader):.4f}")

# 5. Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"test accuracy: {correct / total:.4f}")

# Answers to verbal follow-ups:
# - No softmax: CrossEntropyLoss = log-softmax + NLL in one numerically stable op.
# - Shuffle train: batches should be i.i.d. so gradients aren't biased by data order.
#   Eval just averages over everything, so order can't matter.
# - Overfitting fix: nn.Dropout(0.5) between the layers, weight_decay in Adam,
#   or early stopping on a validation split.
# - model.eval(): disables dropout, BatchNorm uses running stats.
