# PyTorch Cheatsheet — memorize before Tuesday

No AI assist allowed, so this syntax needs to be in your head. Everything here appears in the exercises.

## Tensors

```python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])   # from data
torch.zeros(3, 4); torch.ones(3, 4); torch.rand(3, 4); torch.randn(3, 4)
x.shape          # torch.Size([2, 2])
x.dtype          # torch.float32
x.view(-1)       # reshape (also x.reshape(...)); -1 = infer this dim
x.unsqueeze(0)   # add dim: (2,2) -> (1,2,2)
x @ w            # matrix multiply (or torch.matmul)
x.sum(), x.mean(), x.max(dim=1)
x.argmax(dim=1)  # index of max along dim — used for predictions
```

## Autograd

```python
w = torch.randn(3, requires_grad=True)
loss = (w ** 2).sum()
loss.backward()          # populates w.grad
w.grad                   # dloss/dw
with torch.no_grad():    # inference / manual updates — no graph tracking
    w -= 0.1 * w.grad
w.grad.zero_()           # clear before next backward
```

## Model definition — the template

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),             # (B,1,28,28) -> (B,784)
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),       # logits — no softmax here!
        )

    def forward(self, x):
        return self.net(x)
```

Common layers: `nn.Linear(in, out)`, `nn.Conv2d(in_ch, out_ch, kernel_size, padding)`,
`nn.MaxPool2d(2)`, `nn.ReLU()`, `nn.Dropout(0.5)`, `nn.BatchNorm2d(ch)`, `nn.Flatten()`.

Conv output size: `(W - K + 2P) / S + 1`. With `padding=1, kernel=3, stride=1` size is unchanged; `MaxPool2d(2)` halves it.

## Data

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_ds = datasets.MNIST("./data", train=True, download=True,
                          transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
# each batch: x (64,1,28,28) float, y (64,) int class labels
```

## Loss + optimizer

```python
criterion = nn.CrossEntropyLoss()          # logits + int labels
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## THE training loop (memorize)

```python
model.train()
for epoch in range(3):
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
```

## Evaluation

```python
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"accuracy: {correct / total:.4f}")
```

## Gotchas that come up live

- `CrossEntropyLoss` wants raw logits and integer labels — no softmax, no one-hot.
- Forgot `zero_grad()` → gradients accumulate.
- Forgot `model.eval()` + `torch.no_grad()` at test time → dropout still on, wasted memory.
- `.item()` converts a 1-element tensor to a Python number (for printing/accumulating).
- Shape errors: print `x.shape` — 90% of live-coding bugs are shape mismatches.
