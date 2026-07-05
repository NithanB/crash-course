"""Solution — Exercise 3: CNN on MNIST
Run: python3 ex3_solution.py          (real MNIST, ~3-5 min CPU, >97% acc)
     python3 ex3_solution.py --fake   (tiny synthetic data, just a smoke test)
"""
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

FAKE = "--fake" in sys.argv
transform = transforms.ToTensor()

if FAKE:
    fake = lambda n: datasets.FakeData(size=n, image_size=(1, 28, 28),
                                       num_classes=10, transform=transform)
    train_ds, test_ds = fake(512), fake(256)
else:
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B,16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,16,14,14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B,32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,32,7,7)
            nn.Flatten(),                                 # (B, 32*7*7)
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)

model = CNN()

# Sanity check before training
dummy = torch.rand(4, 1, 28, 28)
print("output shape:", model(dummy).shape, "-> should be (4, 10)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"test accuracy: {correct / total:.4f}")

# Answers to verbal follow-ups:
# - CNN beats MLP with fewer params: weight sharing + locality match image
#   structure; the MLP must relearn each pattern at every pixel position.
# - padding keeps border information / controls output size; stride is the
#   filter's step size (stride 2 downsamples).
# - Dropout usually before the final Linear; BatchNorm2d right after each Conv2d.
