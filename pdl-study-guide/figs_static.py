"""Static concept figures for the PDL Challenge 1 study guide."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
C_TGT, C_RELU, C_TANH, C_SIN, C_PTS = "#666666", "#d62728", "#1f77b4", "#2ca02c", "#111111"
OUT = "/sessions/relaxed-fervent-ptolemy/mnt/outputs/figs"

# ---- Challenge's own generator (np.complex fixed) ----
np.random.seed(42)
N, BL = 512, 50
coeffs = np.zeros(N, dtype=complex)
coeffs[1:BL] = np.random.randn(BL - 1) + 1j * np.random.randn(BL - 1)
coeffs[-(BL - 1):] = coeffs[1:BL][::-1].conj()
f = np.fft.ifft(coeffs).real
n = np.arange(N)

# ================= Fig 1: function + spectrum =================
fig, ax = plt.subplots(1, 2, figsize=(9.5, 2.9))
ax[0].plot(n, f, color=C_TANH, lw=1.2)
ax[0].set(title="Time domain: the band-limited function f[n]", xlabel="sample index n", ylabel="amplitude")
mag = np.abs(coeffs)
ax[1].plot(n, mag, color=C_RELU, lw=0.9)
ax[1].fill_between([0, 49], 0, mag.max() * 1.05, color=C_RELU, alpha=0.08)
ax[1].fill_between([N - 49, N - 1], 0, mag.max() * 1.05, color=C_RELU, alpha=0.08)
ax[1].annotate("band: k = 1…49", xy=(25, mag.max() * 0.95), fontsize=8, color=C_RELU)
ax[1].annotate("mirror band\n(negative freqs)", xy=(N - 120, mag.max() * 0.88), fontsize=8, color=C_RELU)
ax[1].annotate("all zero in between\n→ band-limited", xy=(200, mag.max() * 0.45), fontsize=8, color="#444")
ax[1].set(title="Frequency domain: |F[k]| (only 98 of 512 bins non-zero)", xlabel="frequency bin k", ylabel="|F[k]|")
fig.savefig(f"{OUT}/fig1_bandlimited.png"); plt.close(fig)

# ================= Fig 2: conjugate symmetry =================
ks = np.fft.fftshift(np.fft.fftfreq(N, 1 / N)).astype(int)
csh = np.fft.fftshift(coeffs)
sel = (np.abs(ks) <= 60)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 2.9))
ax[0].stem(ks[sel], np.abs(csh[sel]), linefmt="C0-", markerfmt="C0.", basefmt="k-")
ax[0].set(title="Magnitude |F[k]|: EVEN  (mirror image)", xlabel="frequency k", ylabel="|F[k]|")
phase = np.angle(csh)
phase[np.abs(csh) < 1e-12] = np.nan  # phase undefined where coeff = 0
ax[1].stem(ks[sel], phase[sel], linefmt="C2-", markerfmt="C2.", basefmt="k-")
ax[1].set(title="Phase ∠F[k]: ODD  (sign flips)", xlabel="frequency k", ylabel="phase (rad)")
for a in ax: a.axvline(0, color="k", lw=0.6, alpha=0.5)
fig.suptitle("Conjugate symmetry  F[−k] = F[k]*  ⇔  f[n] is real-valued", y=1.04, fontsize=10)
fig.savefig(f"{OUT}/fig2_symmetry.png"); plt.close(fig)

# ================= Fig 3: enough vs too few samples =================
def design(idx):
    k = np.arange(1, BL)
    ang = 2 * np.pi * np.outer(idx, k) / N
    return np.hstack([np.cos(ang), np.sin(ang)])  # (len(idx), 98)

def linrec(idx):
    A = design(idx)
    c, *_ = np.linalg.lstsq(A, f[idx], rcond=None)
    return design(n) @ c

idx128, idx64 = np.arange(0, N, 4), np.arange(0, N, 8)
rec128, rec64 = linrec(idx128), linrec(idx64)
mse128, mse64 = np.mean((rec128 - f) ** 2), np.mean((rec64 - f) ** 2)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0), sharey=True)
for a, idx, rec, mse, ttl in [
    (ax[0], idx128, rec128, mse128, "M = 128 regular samples  (≥ 98 needed)  →  EXACT"),
    (ax[1], idx64, rec64, mse64, "M = 64 regular samples  (< 98)  →  ALIASING"),
]:
    a.plot(n, f, color=C_TGT, lw=2.2, alpha=0.45, label="true f[n]")
    a.plot(n, rec, color=C_RELU, lw=1.0, label="linear (least-squares) reconstruction")
    a.plot(idx, f[idx], ".", color=C_PTS, ms=3, label="samples")
    a.set(title=ttl + f"\nreconstruction MSE = {mse:.1e}", xlabel="n")
    a.set_xlim(0, 511)
ax[0].set_ylabel("amplitude"); ax[0].legend(fontsize=7, loc="upper right")
fig.savefig(f"{OUT}/fig3_nyquist.png"); plt.close(fig)
print("fig3 MSEs:", mse128, mse64)

# ================= Fig 4: activation gallery =================
x = np.linspace(-4, 4, 400)
acts = [
    ("ReLU", np.maximum(0, x), (x > 0).astype(float), "cheap, sharp corner;\n'dead' for x<0"),
    ("Leaky ReLU (0.1)", np.where(x > 0, x, 0.1 * x), np.where(x > 0, 1, 0.1), "no dead neurons,\nstill piecewise-linear"),
    ("Sigmoid", 1 / (1 + np.exp(-x)), None, "smooth; saturates both\nsides → slow learning"),
    ("Tanh", np.tanh(x), 1 - np.tanh(x) ** 2, "smooth, zero-centred;\nsaturates for |x|>2"),
    ("GELU", 0.5 * x * (1 + np.vectorize(np.math.erf if hasattr(np, 'math') else None)(x)) if False else None, None, ""),
    ("sin(x)", np.sin(x), np.cos(x), "periodic, never saturates;\nbasis of SIREN networks"),
]
from math import erf
gelu = 0.5 * x * (1 + np.array([erf(v / np.sqrt(2)) for v in x]))
sig = 1 / (1 + np.exp(-x))
acts[2] = ("Sigmoid", sig, sig * (1 - sig), "smooth; saturates both\nsides → slow learning")
dgelu = np.gradient(gelu, x)
acts[4] = ("GELU", gelu, dgelu, "smooth ReLU; default in\nmodern transformers")
fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2))
for a, (name, y, dy, note) in zip(axes.flat, acts):
    a.plot(x, y, color=C_TANH, lw=1.6, label="f(x)")
    if dy is not None: a.plot(x, dy, color=C_RELU, lw=1.1, ls="--", label="f ′(x)")
    a.set_title(name); a.axhline(0, color="k", lw=0.4); a.axvline(0, color="k", lw=0.4)
    a.text(0.03, 0.02, note, transform=a.transAxes, fontsize=7, color="#444", va="bottom")
    a.set_ylim(-1.6, max(2.0, np.nanmax(y) * 0.6))
axes[0, 0].legend(fontsize=7, loc="upper left")
fig.suptitle("Activation functions and their derivatives — the derivative shapes what gradients (and fits) look like", y=1.01, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_activations.png"); plt.close(fig)

# ================= Fig 5: task architecture diagrams =================
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
def draw_net(a, layers, title, budget, note):
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off"); a.grid(False)
    xs = np.linspace(1, 9, len(layers))
    for i, ((label, sz), xc) in enumerate(zip(layers, xs)):
        h = 1.6 + 4.5 * (np.log10(sz) / 3.0)
        r = Rectangle((xc - 0.62, 5 - h / 2), 1.24, h, fc="#dbe9f6" if 0 < i < len(layers) - 1 else "#f6e3db",
                      ec="#345", lw=1)
        a.add_patch(r)
        a.text(xc, 5, f"{label}\n({sz})", ha="center", va="center", fontsize=7.5)
        if i < len(layers) - 1:
            a.annotate("", xy=(xs[i + 1] - 0.66, 5), xytext=(xc + 0.66, 5),
                       arrowprops=dict(arrowstyle="->", color="#345", lw=1))
    a.set_title(title, fontsize=10)
    a.text(5, 0.55, budget, ha="center", fontsize=8.5, color="#a33")
    a.text(5, 9.4, note, ha="center", fontsize=8, color="#444")
draw_net(axes[0], [("input x", 1), ("hidden", 128), ("hidden", 128), ("hidden", 128), ("output y", 1)],
         "Task 1: memorise ONE function", "budget: ≤ 10,000 neurons total (e.g. 3×128 = 384 used)",
         "scalar in → scalar out;  trained on (x, y) pairs of one fixed f")
draw_net(axes[1], [("100 samples\nf(X)", 100), ("hidden", 2048), ("hidden", 2048), ("full function\nf[1…512]", 512)],
         "Task 2: learn the reconstruction OPERATOR", "budget: ≤ 1,000,000 neurons (e.g. 2×2048 = 4096 used)",
         "vector in → vector out;  trained on MANY random functions")
fig.savefig(f"{OUT}/fig5_architectures.png"); plt.close(fig)

# ================= Fig 6: regular vs irregular X =================
rng = np.random.default_rng(0)
reg = np.linspace(0, N - 1, 100).astype(int)
irr = np.sort(rng.choice(N, 100, replace=False))
expo = np.unique(np.round(np.logspace(0, np.log10(N - 1), 100)).astype(int))
fig, ax = plt.subplots(3, 1, figsize=(9.5, 4.6), sharex=True)
for a, idx, ttl, col in [(ax[0], reg, f"REGULAR: 100 uniformly spaced (gap ≈ 5.1)", C_TANH),
                          (ax[1], irr, "IRREGULAR (random): 100 points — uneven gaps", C_SIN),
                          (ax[2], expo, f"IRREGULAR (exponential): {len(expo)} unique points — huge gaps on the right", C_RELU)]:
    a.plot(n, f, color=C_TGT, lw=1.0, alpha=0.5)
    a.plot(idx, f[idx], ".", color=col, ms=4)
    gaps = np.diff(idx)
    a.set_title(ttl + f"   (largest gap = {gaps.max()} samples)", fontsize=9)
ax[2].set_xlabel("sample index n")
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_sampling_patterns.png"); plt.close(fig)
print("expo unique:", len(expo), "max gaps:", np.diff(reg).max(), np.diff(irr).max(), np.diff(expo).max())
print("static figs done")
