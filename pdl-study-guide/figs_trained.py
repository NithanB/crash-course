"""Trained-network figures: activation comparison, spectral bias, dataset size."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
C_TGT, C_RELU, C_TANH, C_SIN, C_PTS = "#666666", "#d62728", "#1f77b4", "#2ca02c", "#111111"
OUT = "/sessions/relaxed-fervent-ptolemy/mnt/outputs/figs"

# target function (same as challenge, seed 42)
np.random.seed(42)
N, BL = 512, 50
coeffs = np.zeros(N, dtype=complex)
coeffs[1:BL] = np.random.randn(BL - 1) + 1j * np.random.randn(BL - 1)
coeffs[-(BL - 1):] = coeffs[1:BL][::-1].conj()
f = np.fft.ifft(coeffs).real
n = np.arange(N)
x_all = (2 * n / (N - 1) - 1).reshape(-1, 1)          # normalise x to [-1, 1]
y_mu, y_sd = f.mean(), f.std()
y_all = ((f - y_mu) / y_sd).reshape(-1, 1)            # standardise y

# ---------------- tiny numpy MLP with Adam ----------------
class MLP:
    def __init__(self, sizes, act, seed=0, w0=30.0):
        g = np.random.default_rng(seed)
        self.act, self.w0 = act, w0
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            if act == "sin":                      # SIREN init (Sitzmann et al. 2020)
                lim = 1 / a if i == 0 else np.sqrt(6 / a) / w0
                W = g.uniform(-lim, lim, (a, b))
            elif act == "relu":
                W = g.normal(0, np.sqrt(2 / a), (a, b))
            else:                                  # tanh: Xavier
                W = g.uniform(-np.sqrt(6 / (a + b)), np.sqrt(6 / (a + b)), (a, b))
            self.W.append(W); self.b.append(np.zeros(b))
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X):
        self.zs, self.as_ = [], [X]
        a = X
        L = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            self.zs.append(z)
            if i == L - 1:
                a = z                              # linear output
            elif self.act == "relu":
                a = np.maximum(0, z)
            elif self.act == "tanh":
                a = np.tanh(z)
            else:
                a = np.sin(self.w0 * z)
            self.as_.append(a)
        return a

    def step(self, X, Y, lr):
        P = self.forward(X)
        m = X.shape[0]
        delta = 2 * (P - Y) / m
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        L = len(self.W)
        for i in reversed(range(L)):
            gW[i] = self.as_[i].T @ delta
            gb[i] = delta.sum(0)
            if i > 0:
                da = delta @ self.W[i].T
                z = self.zs[i - 1]
                if self.act == "relu":
                    delta = da * (z > 0)
                elif self.act == "tanh":
                    delta = da * (1 - np.tanh(z) ** 2)
                else:
                    delta = da * self.w0 * np.cos(self.w0 * z)
        self.t += 1
        params = self.W + self.b
        grads = gW + gb
        for j, (p, g) in enumerate(zip(params, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            mh = self.m[j] / (1 - 0.9 ** self.t)
            vh = self.v[j] / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return np.mean((P - Y) ** 2)

def train(act, idx, iters, lr0, seed=0, sizes=(1, 128, 128, 128, 1)):
    net = MLP(sizes, act, seed=seed)
    X, Y = x_all[idx], y_all[idx]
    for i in range(iters):
        lr = lr0 if i < iters * 0.6 else lr0 / 5
        tr = net.step(X, Y, lr)
    pred = net.forward(x_all)
    return pred.ravel(), float(tr), float(np.mean((pred - y_all) ** 2))

rng = np.random.default_rng(7)
idx100 = np.sort(rng.choice(N, 100, replace=False))

JOBS = {
    "relu": ("relu", None, 25000, 1e-3),
    "tanh": ("tanh", None, 25000, 1e-3),
    "sin": ("sin", None, 8000, 1e-4),
    "M16": ("tanh", 16, 20000, 1e-3),
    "M64": ("tanh", 64, 20000, 1e-3),
    "M256": ("tanh", 256, 20000, 1e-3),
}

import sys, os
mode = sys.argv[1]

if mode in JOBS:
    act, M, it, lr = JOBS[mode]
    idx = idx100 if M is None else np.sort(np.random.default_rng(3).choice(N, M, replace=False))
    p, tr, te = train(act, idx, it, lr)
    np.savez(f"{OUT}/run_{mode}.npz", pred=p, idx=idx, train_mse=tr, test_mse=te)
    print(mode, tr, te, flush=True)
    sys.exit()

# ---------------- mode == "plot" ----------------
# least-squares baseline that KNOWS the band-limit
kk = np.arange(1, BL)
def design(idx):
    ang = 2 * np.pi * np.outer(idx, kk) / N
    return np.hstack([np.cos(ang), np.sin(ang)])
c_ls, _, rank, _ = np.linalg.lstsq(design(idx100), y_all.ravel()[idx100], rcond=None)
rec_ls = design(n) @ c_ls
meta = {"least_squares": {"test_mse": float(np.mean((rec_ls - y_all.ravel()) ** 2)), "rank": int(rank)}}
preds, idxs = {}, {}
for name in JOBS:
    d = np.load(f"{OUT}/run_{name}.npz")
    preds[name], idxs[name] = d["pred"], d["idx"]
    meta[name] = {"train_mse": float(d["train_mse"]), "test_mse": float(d["test_mse"])}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)

# ================= Fig 7: same data, three activations =================
fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True)
for a, (act, col, name) in zip(axes, [("relu", C_RELU, "ReLU"), ("tanh", C_TANH, "Tanh"), ("sin", C_SIN, "Sine (SIREN)")]):
    a.plot(n, y_all.ravel(), color=C_TGT, lw=2.2, alpha=0.45, label="true function")
    a.plot(n, preds[act], color=col, lw=1.1, label=f"{name} fit")
    a.plot(idx100, y_all[idx100], ".", color=C_PTS, ms=4, label="100 training points")
    m = meta[act]
    a.set_title(f"{name}  —  train MSE {m['train_mse']:.1e},   full-grid MSE {m['test_mse']:.2f}", color=col)
    a.legend(fontsize=7, loc="upper right", ncol=3)
axes[-1].set_xlabel("sample index n")
fig.suptitle("Same 100 training points, same 1→128→128→128→1 net — only the activation changes\n"
             f"(for reference: band-limit-aware least squares on the SAME points is exact, MSE ≈ {meta['least_squares']['test_mse']:.0e})",
             y=1.02, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_activation_fits.png"); plt.close(fig)

# ================= Fig 8: spectral bias =================
fig, ax = plt.subplots(figsize=(9.5, 3.0))
tgt_mag = np.abs(np.fft.rfft(y_all.ravel()))
ax.plot(tgt_mag, color=C_TGT, lw=2.6, alpha=0.5, label="true spectrum")
for act, col, name in [("relu", C_RELU, "ReLU"), ("tanh", C_TANH, "Tanh"), ("sin", C_SIN, "Sine (SIREN)")]:
    ax.plot(np.abs(np.fft.rfft(preds[act])), color=col, lw=1.0, label=name)
ax.axvline(49, color="k", lw=0.8, ls=":")
ax.text(50.5, tgt_mag.max() * 0.9, "band edge k=49", fontsize=8)
ax.set(xlim=(0, 90), xlabel="frequency bin k", ylabel="|FFT of fit|",
       title="Spectral bias: low frequencies are learned; high frequencies near the band edge are missed")
ax.legend(fontsize=8)
fig.savefig(f"{OUT}/fig8_spectral_bias.png"); plt.close(fig)

# ================= Fig 9: dataset size =================
fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True)
for a, M in zip(axes, [16, 64, 256]):
    m = meta[f"M{M}"]
    idx = idxs[f"M{M}"]
    a.plot(n, y_all.ravel(), color=C_TGT, lw=2.2, alpha=0.45, label="true function")
    a.plot(n, preds[f"M{M}"], color=C_TANH, lw=1.1, label="tanh fit")
    a.plot(idx, y_all[idx], ".", color=C_PTS, ms=4, label="training points")
    tag = "(M < 98: the data can't pin the function down)" if M < 98 else "(M > 98: enough information now)"
    a.set_title(f"M = {M} points  —  train MSE {m['train_mse']:.1e},   full-grid MSE {m['test_mse']:.2f}   {tag}")
    a.legend(fontsize=7, loc="upper right", ncol=3)
axes[-1].set_xlabel("sample index n")
fig.suptitle("Tanh network, identical architecture — only the dataset size M changes", y=1.005, fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_dataset_size.png"); plt.close(fig)
print("DONE")
