

"""
Generate target band-limited function and plot it
"""

# TODO for your experiments:
    #  - sweep M in [10, 25, 50, 100, 200, 400]; plot full-grid MSE vs M (log-log)
    #  - sweep depth/width at fixed neuron budget (e.g. 1x384 vs 3x128 vs 6x64)
    #  - compare mode="regular" vs "random" vs "exponential"
    #  - repeat each config over >=3 seeds; report mean +/- std

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def bandlimited_function(N=512, band_limit = 50, seed = 42):
    
    rng = np.random.RandomState(seed)
    c = np.zeros(N, dtype=complex)                      # np.complex is removed in NumPy>=1.24
    c[1:band_limit] = rng.randn(band_limit - 1) + 1j * rng.randn(band_limit - 1)
    c[-(band_limit - 1):] = c[1:band_limit][::-1].conj() # conjugate symmetry
    return np.fft.ifft(c).real                          # shape (N,)




N = 512 
f = bandlimited_function(N)



"""
create dataset of M points of (x, y) and plot it
"""

def make_dataset(f, M , mode = "random", seed = 0):
    
    N = len(f)
    
    if mode == "regular":
        idx = np.linspace(0, N - 1, M).astype(int)
        
    elif mode == "random":
        idx = np.sort(np.random.default_rng(seed).choice(N, M, replace=False))
        
    elif mode == "exponential":
        idx = np.unique(np.round(np.logspace(0, np.log10(N - 1), M)).astype(int))
    
    x = 2 * idx / (N - 1) - 1
    y = (f[idx]-f.mean()) / f.std()
    
    return (torch.tensor(x, dtype = torch.float32). unsqueeze(1),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1), idx)
    
    
    
"""
Build the Model!
"""

class Sine(nn.Module):
    def __init__(self,  w0 = 30.0): super().__init__(); self.w0 = w0
    def forward(self, x): return torch.sin(self.w0 * x)
    
def make_mlp(depth = 3, width = 128, act = "tanh"):
    acts = {"relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
            "sine": Sine}
    
    layers, d_in = [], 1
    
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), acts[act]()]
        d_in = width
    
    layers += [nn.Linear(d_in, 1)]
    net = nn.Sequential(*layers)
    
    n_neurons = depth * width
    
    assert n_neurons <= 10_000, f"budget exceeded: {n_neurons}"
    
    print(f"{depth}x{width} {act}: {n_neurons} neurons, "
          f"{sum(p.numel() for p in net.parameters())} parameters")
    
    return net


"""
Train the model until convergence
"""

def train(net, X, Y, lr = 1e-3, max_iters = 30_000, tol=1e-7):
    
    opt = torch.optim.Adam(net.parameters(), lr = lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.5, patience = 2000)
    
    prev = np.inf
    
    for i in range(max_iters):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(X), Y)
        loss.backward()
        opt.step()
        sched.step(loss)
        if i% 2000 == 0:
            print(f"  iter {i:6d}  train MSE {loss.item():.3e}")
            if abs(prev - loss.item()) < tol: break
            prev = loss.item()
    return loss.item()



"""
evaluate on FULL Grid (TEST)
"""

def evaluate(net, f):
    N = len(f)
    x_full = torch.linspace(-1, 1, N).unsqueeze(1)
    
    with torch.no_grad():
        pred = net(x_full).squeeze().numpy()
        
    y_full = (f - f.mean()) / f.std()
        
    return pred, float(np.mean((pred - y_full) ** 2))

"""Experiment Loop
"""

def sweep_M():
    torch.manual_seed(0)                     # reproducible init each run

    Ms = [10, 25, 50, 100, 200, 400]
    mses = []

    for M in Ms:
        X, Y, idx = make_dataset(f, M, mode="random")
        net = make_mlp(depth=3, width=128, act="tanh")
        tr = train(net, X, Y, lr=1e-3)
        pred, test_mse = evaluate(net, f)
        mses.append(test_mse)
        print(f"M = {M}: train {tr:.2e}   full-grid {test_mse:.3f}\n")

    # --- plot full-grid MSE vs M on log-log axes ---
    plt.figure()
    plt.loglog(Ms, mses, "o-")
    plt.xlabel("M (number of training points)")
    plt.ylabel("full-grid MSE")
    plt.title("Full-grid MSE vs M")
    plt.grid(True, which="both")             # major + minor gridlines
    plt.savefig("mse_vs_M.png", dpi=150)
    plt.show()

    return Ms, mses





def compare_activations():
    torch.manual_seed(0)
    results = {}
    M = 100
    X, Y, idx = make_dataset(f, M, mode="random")
    for act in ["tanh"]:
        net = make_mlp(depth=3, width=128, act=act)
        tr = train(net, X, Y, lr=1e-4 if act == "tanh" else 1e-3)
        pred, test_mse = evaluate(net, f)
        results[act] = (pred, tr, test_mse)
        print(f"==> {act}: train {tr:.2e}   full-grid {test_mse:.3f}\n")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    y_full = (f - f.mean()) / f.std()
    for ax, (act, (pred, tr, te)) in zip(axes, results.items()):
        ax.plot(y_full, "gray", lw=2, alpha=0.5, label="true")
        ax.plot(pred, lw=1, label=f"{act} (MSE {te:.3f})")
        ax.plot(idx, y_full[idx], "k.", ms=4)
        ax.legend()
    plt.savefig("task1_results.png", dpi=150); plt.show()


def tanh_fit_vs_M(Ms=(8, 64,    512)):
    """Fit a tanh network at each M and show its reconstruction in a SEPARATE window."""
    y_full = (f - f.mean()) / f.std()

    for M in Ms:
        torch.manual_seed(0)                             # same init each M -> fair comparison
        X, Y, idx = make_dataset(f, M, mode="random")
        net = make_mlp(depth=3, width=128, act="tanh")
        tr = train(net, X, Y, lr=1e-3)
        pred, test_mse = evaluate(net, f)
        print(f"M = {M}: train {tr:.2e}   full-grid {test_mse:.3f}\n")

        plt.figure(figsize=(10, 4))                      # new window for this M
        plt.plot(y_full, "gray", lw=2, alpha=0.5, label="true")
        plt.plot(pred, "C0", lw=1, label=f"tanh fit (MSE {test_mse:.3f})")
        plt.plot(idx, y_full[idx], "k.", ms=6, label=f"{M} samples")
        plt.title(f"tanh network — M = {M}")
        plt.legend()
        plt.savefig(f"task1_tanh_M{M}.png", dpi=150)

    plt.show()                                           # opens all the windows at once


if __name__ == "__main__":
    tanh_fit_vs_M()
    # sweep_M()               # log-log MSE-vs-M plot
    # compare_activations()   # relu/tanh/sine comparison
    