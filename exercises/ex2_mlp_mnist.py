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


datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
# ---------------------------------------------------------------
# 1. Data: MNIST train and test sets, with ToTensor transform.
#    Wrap each in a DataLoader (batch_size=64, shuffle train only).
# ---------------------------------------------------------------
transform = transforms.ToTensor()

train_ds = datasets.MNIST(root="./data", train=True, download = True, transform=transform)  # TODO: datasets.MNIST(...)
test_ds = datasets.MNIST(root="./data", train=False, download = True, transform=transform)   # TODO

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)  # TODO
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)   # TODO

# ---------------------------------------------------------------
# 2. Model: an MLP as an nn.Module subclass.
#    784 -> 128 -> ReLU -> 10 (logits, no softmax — know why!)
# ---------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), #nn.flatten first
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
            
        )
        

    def forward(self, x):
        return self.net(x)

model = MLP()

# ---------------------------------------------------------------
# 3. Loss and optimizer: cross-entropy + Adam (lr=1e-3).
# ---------------------------------------------------------------
criterion = nn.CrossEntropyLoss()  # TODO
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # TODO

# ---------------------------------------------------------------
# 4. Training loop: 2 epochs. The 5 sacred steps per batch:
#    zero_grad -> forward -> loss -> backward -> step
#    Print average loss per epoch.
# ---------------------------------------------------------------
for epoch in range(2):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        

    print(f"epoch {epoch + 1}: avg loss {total_loss / len(train_loader):.4f}")

# ---------------------------------------------------------------
# 5. Evaluation: accuracy on the test set.
#    Remember: model.eval() and torch.no_grad().
#    Prediction = argmax over the 10 logits.
# ---------------------------------------------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x,y in test_loader:
        preds = model(x).argmax(dim = 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        






print(f"test accuracy: {correct / total:.4f}")  # want > 0.95

# ---------------------------------------------------------------
# Verbal follow-ups to be able to answer:
# - Why no softmax in the model?
# - Why shuffle the training data but not the test data?
# - What would you change to reduce overfitting? (add nn.Dropout)
# - What does model.eval() change?
# ---------------------------------------------------------------
