"""
PDL Challenge 1 — Task 2: a DNN learns to reconstruct ALL band-limited
functions from 100 samples.  Input: 100 values f(X).  Output: all 512 values.
Requires: pip install torch numpy matplotlib
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N, BAND = 256, 50

# ---------------- 1. Generate M random band-limited functions ----------------
def random_functions(M, seed=0):
    rng = np.random.RandomState(seed)
    F = np.zeros((M, N))
    for m in range(M):
        c = np.zeros(N, dtype=complex)
        c[1:BAND] = rng.randn(BAND - 1) + 1j * rng.randn(BAND - 1)
        c[-(BAND - 1):] = c[1:BAND][::-1].conj()
        F[m] = np.fft.ifft(c).real
    return F                                             # (M, 512)

# ---------------- 2. Choose the sampling set X (100 indexes) ----------------
def sampling_set(kind="regular", n_samples=100, seed=0):
    if kind == "regular":
        return np.linspace(0, N - 1, n_samples).astype(int)
    if kind == "random":
        return np.sort(np.random.default_rng(seed).choice(N, n_samples, replace=False))
    if kind == "exponential":
        return np.unique(np.round(np.logspace(0, np.log10(N - 1), n_samples)).astype(int))

# ---------------- 3. Dataset: predictor f(X) (M,100) -> label f (M,512) ----------------
def make_dataset(M, kind="regular", seed=0):
    F = random_functions(M, seed)
    Xset = sampling_set(kind)
    scale = F.std()                                      # one global scale
    A = torch.tensor(F[:, Xset] / scale, dtype=torch.float32)
    B = torch.tensor(F / scale, dtype=torch.float32)
    return A, B, Xset, scale

# ---------------- 4. Model (stay under 1,000,000 neurons) ----------------
def make_model(depth=2, width=2048, act="relu", n_in=100):
    acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "linear": None}
    layers, d = [], n_in
    for _ in range(depth if act != "linear" else 0):
        layers += [nn.Linear(d, width), acts[act]()]
        d = width
    layers += [nn.Linear(d, N)]
    net = nn.Sequential(*layers)
    print(f"{act} {depth}x{width}: {depth*width} neurons, "
          f"{sum(p.numel() for p in net.parameters()):,} parameters")
    return net

# ---------------- 5. Train / evaluate ----------------
def train(net, A, B, Aval, Bval, lr=1e-3, epochs=3000, batch=256):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    M = A.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(M)
        for i in range(0, M, batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(A[j]), B[j])
            loss.backward(); opt.step()
        if ep % 200 == 0:
            with torch.no_grad():
                vl = nn.functional.mse_loss(net(Aval), Bval).item()
            print(f"  epoch {ep:5d}  train {loss.item():.3e}  VAL {vl:.3e}")
    # final validation MSE (used by the sweeps below)
    with torch.no_grad():
        final_val = nn.functional.mse_loss(net(Aval), Bval).item()
    return net, final_val


# ---------------- Experiment 1: sweep M, plot VAL MSE vs M (log-log) ----------------
def sweep_M():
    torch.manual_seed(0)

    # a FIXED validation set of brand-new functions, reused across every M
    Aval, Bval, _, _ = make_dataset(500, kind="regular", seed=999)

    Ms = [100, 500, 1000]
    val_mses = []

    for M in Ms:
        A, B, Xset, scale = make_dataset(M, kind="regular", seed=0)
        net = make_model(depth=2, width=2048, act="relu")
        net, final_val = train(net, A, B, Aval, Bval)
        val_mses.append(final_val)
        print(f"M = {M:6d}:  VAL MSE {final_val:.3e}\n")

    # --- plot VAL MSE vs M on log-log axes ---
    plt.figure()
    plt.loglog(Ms, val_mses, "o-")
    plt.xlabel("M (number of training functions)")
    plt.ylabel("validation MSE")
    plt.title("VAL MSE vs training-set size M")
    plt.grid(True, which="both")
    plt.savefig("task2_mse_vs_M.png", dpi=150)
    plt.show()

    return Ms, val_mses


if __name__ == "__main__":
    sweep_M()
