"""
Exercise 3: CNN on MNIST (~60 min)
Same pipeline as Exercise 2 — only the model changes. This teaches you
that in PyTorch, swapping architectures is swapping one class.

Fill in the TODOs, run:  python3 ex3_cnn.py
Target: >97% test accuracy after 2 epochs (~3-5 min on CPU).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

# ---------------------------------------------------------------
# 1. The CNN. Target architecture:
#    Conv2d(1 -> 16, kernel 3, padding 1) -> ReLU -> MaxPool2d(2)   # 28x28 -> 14x14
#    Conv2d(16 -> 32, kernel 3, padding 1) -> ReLU -> MaxPool2d(2)  # 14x14 -> 7x7
#    Flatten -> Linear(32*7*7 -> 10)
#
#    BEFORE coding, verify on paper why the Linear input is 32*7*7.
#    (padding=1 with kernel 3 preserves size; each pool halves it)
# ---------------------------------------------------------------

lin_input = 32 * 7 * 7  # TODO: compute this from the conv/pool layers above
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnet = nn.Sequential(
            nn.Conv2d(1,16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(lin_input, 10)
        )

    def forward(self, x):
        return self.mnet(x)

model = CNN()

# Sanity check BEFORE training (do this habitually in the live session):
dummy = torch.rand(4, 1, 28, 28)
out = model(dummy)
print("output shape:", out.shape, "-> should be (4, 10)")

# ---------------------------------------------------------------
# 2. Loss, optimizer, training loop, evaluation.
#    Same as Exercise 2 — write it again WITHOUT looking at ex2.
#    Repetition is the goal.
# ---------------------------------------------------------------
criterion = nn.CrossEntropyLoss()  

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3 )

for epoch in range(2):
    model.train()
    total_loss = 0.0
    for x,y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch + 1}: avg loss {total_loss / len(train_loader):.4f}")

model.eval()
correct, total = 0, 0
#evaluation loop

with torch.no_grad():
    for x,y in test_loader:
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"test accuracy: {correct / total:.4f}")  # want > 0.97

# ---------------------------------------------------------------
# Verbal follow-ups to be able to answer:
# - Why does the CNN beat the MLP with fewer parameters?
# - What do padding and stride do?
# - Where would you add Dropout or BatchNorm here?
# - Stretch: add nn.BatchNorm2d after each conv and see the effect.
# ---------------------------------------------------------------
