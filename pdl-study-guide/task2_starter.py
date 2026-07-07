"""
PDL Challenge 1 — Task 2 starter: a DNN learns to reconstruct ALL band-limited
functions from 100 samples.  Input: 100 values f(X).  Output: all 512 values.
Requires: pip install torch numpy matplotlib
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N, BAND = 512, 50

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
    return net

if __name__ == "__main__":
    torch.manual_seed(0)
    M = 2000                                             # sweep this: 100 ... 20000
    A, B, Xset, scale = make_dataset(M, kind="regular", seed=0)
    Aval, Bval, _, _ = make_dataset(500, kind="regular", seed=999)   # NEW functions!

    net = make_model(depth=2, width=2048, act="relu")
    train(net, A, B, Aval, Bval)

    # --- linear baseline: 100 regular samples >= 98 degrees of freedom => exact ---
    # a purely linear model (act="linear") can reach ~0 error; can your DNN beat it?

    with torch.no_grad():
        rec = net(Aval[:3]).numpy() * scale
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(Bval[k].numpy() * scale, "gray", lw=2, alpha=0.5, label="true (unseen function)")
        ax.plot(rec[k], "C3", lw=1, label="DNN reconstruction")
        ax.plot(Xset, Aval[k].numpy() * scale, "k.", ms=4, label="100 given samples")
        ax.legend()
    plt.savefig("task2_results.png", dpi=150); plt.show()

    # TODO for your experiments:
    #  - sweep dataset size M; plot VAL MSE vs M (log-log). Where does it plateau?
    #  - act="linear" vs "relu" vs "tanh": does nonlinearity even help here? why?
    #  - kind="regular" vs "random" vs "exponential": where do errors concentrate?
    #    (plot per-position error averaged over the val set vs sample locations)
    #  - depth vs width under the same neuron budget
