# Environment Setup — Mac (Apple Silicon)

Do this **today**, not the morning of the meeting. Budget 30 minutes.

## 1. Python + virtual environment

macOS ships with Python 3. Check version (need 3.9+):

```bash
python3 --version
```

Create a project folder and virtual environment:

```bash
mkdir -p ~/dl-meeting && cd ~/dl-meeting
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`.

## 2. Install PyTorch

```bash
pip install --upgrade pip
pip install torch torchvision
```

On Apple Silicon this installs PyTorch with **MPS** (Metal GPU) support automatically — no CUDA needed.

## 3. Verify

```bash
python3 -c "import torch; print(torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

Expect a version number and `MPS available: True`. For the meeting, **just use CPU** (`device = "cpu"`) — MNIST-scale exercises run in seconds and you avoid MPS edge cases while screen sharing.

## 4. VS Code

1. Install VS Code: https://code.visualstudio.com
2. Install extensions: **Python** and **Jupyter** (both by Microsoft).
3. Open the folder: `code ~/dl-meeting` (or File → Open Folder).
4. `Cmd+Shift+P` → "Python: Select Interpreter" → pick `.venv/bin/python`.
5. Test: create `test.py` with `import torch; print(torch.rand(2,3))`, run with the ▶ button or `python3 test.py` in the integrated terminal (`` Ctrl+` ``).

## 5. Turn OFF AI assistance (important)

They'll watch you code without AI help. Remove anything that could autocomplete answers:

- Disable/uninstall **GitHub Copilot** and any AI extensions (Extensions panel → gear → Disable).
- Settings (`Cmd+,`) → search "inline suggest" → uncheck **Editor › Inline Suggestions** if you want zero ghost text.
- Basic IntelliSense (function name completion) is normal and fine to keep.

## 6. Pre-download MNIST (avoid network delays live)

Run once today so the dataset is cached:

```python
from torchvision import datasets, transforms
datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
```

## 7. Screen-share dry run

Before Tuesday: open your meeting app (Zoom/Meet), share your screen with VS Code open, run a script, and confirm font size is readable (`Cmd+=` to zoom in VS Code).
