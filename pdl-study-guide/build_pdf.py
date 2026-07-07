# Build the PDL Challenge 1 study guide PDF (reportlab / platypus)
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, Table, TableStyle, PageBreak,
                                KeepTogether, Preformatted, HRFlowable)

F = "/usr/share/fonts/truetype/dejavu/"
for name, file in [("DVS", "DejaVuSans.ttf"), ("DVS-B", "DejaVuSans-Bold.ttf"),
                   ("DVS-I", "DejaVuSans-Oblique.ttf"), ("DVS-BI", "DejaVuSans-BoldOblique.ttf"),
                   ("DVM", "DejaVuSansMono.ttf"), ("DVM-B", "DejaVuSansMono-Bold.ttf")]:
    pdfmetrics.registerFont(TTFont(name, F + file))
pdfmetrics.registerFontFamily("DVS", normal="DVS", bold="DVS-B", italic="DVS-I", boldItalic="DVS-BI")

OUT = "/sessions/relaxed-fervent-ptolemy/mnt/outputs"
FIG = OUT + "/figs"
PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
FRAME_W = PAGE_W - 2 * MARGIN

NAVY = colors.HexColor("#173a5e")
BLUE = colors.HexColor("#1f77b4")
LIGHT = colors.HexColor("#eef4fa")
AMBER = colors.HexColor("#fdf3e0")
AMBER_EDGE = colors.HexColor("#e0a53c")
GRAY = colors.HexColor("#555555")
CODE_BG = colors.HexColor("#f5f5f2")

S = {
    "title": ParagraphStyle("title", fontName="DVS-B", fontSize=19, leading=24, textColor=NAVY),
    "subtitle": ParagraphStyle("subtitle", fontName="DVS", fontSize=11, leading=15, textColor=GRAY),
    "h1": ParagraphStyle("h1", fontName="DVS-B", fontSize=13.5, leading=17, textColor=NAVY,
                          spaceBefore=14, spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DVS-B", fontSize=10.8, leading=14, textColor=BLUE,
                          spaceBefore=10, spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DVS", fontSize=9.2, leading=13.2, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="DVS", fontSize=9.2, leading=13.2, spaceAfter=3,
                              leftIndent=14, bulletIndent=4),
    "caption": ParagraphStyle("caption", fontName="DVS-I", fontSize=8.1, leading=10.5,
                               textColor=GRAY, spaceBefore=2, spaceAfter=10, alignment=TA_CENTER),
    "boxtitle": ParagraphStyle("boxtitle", fontName="DVS-B", fontSize=9.2, leading=12, textColor=NAVY),
    "boxbody": ParagraphStyle("boxbody", fontName="DVS", fontSize=9.0, leading=12.6),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=6.9, leading=8.6),
    "tcell": ParagraphStyle("tcell", fontName="DVS", fontSize=8.4, leading=11),
    "tcellb": ParagraphStyle("tcellb", fontName="DVS-B", fontSize=8.4, leading=11, textColor=NAVY),
    "qa_q": ParagraphStyle("qa_q", fontName="DVS-B", fontSize=9.0, leading=12.4, spaceBefore=5),
    "qa_a": ParagraphStyle("qa_a", fontName="DVS", fontSize=9.0, leading=12.4, leftIndent=12,
                            textColor=colors.HexColor("#333333")),
}

story = []
def P(text, style="body"): story.append(Paragraph(text, S[style]))
def B(text): story.append(Paragraph(text, S["bullet"], bulletText="•"))
def SP(h=6): story.append(Spacer(1, h))

def fig(path, caption, width=FRAME_W):
    im = PILImage.open(path)
    w, h = im.size
    width = min(width, FRAME_W)
    img = Image(path, width=width, height=width * h / w)
    story.append(KeepTogether([img, Paragraph(caption, S["caption"])]))

def box(title, text, bg=LIGHT, edge=BLUE):
    inner = [Paragraph(title, S["boxtitle"]), Spacer(1, 2), Paragraph(text, S["boxbody"])]
    t = Table([[inner]], colWidths=[FRAME_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.8, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(KeepTogether(t)); SP(8)

def table(header, rows, colw, style_extra=()):
    data = [[Paragraph(h, S["tcellb"]) for h in header]] + \
           [[Paragraph(c, S["tcell"]) for c in row] for row in rows]
    t = Table(data, colWidths=colw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c6d4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        *style_extra]))
    story.append(t); SP(8)

# ============================ COVER / INTRO ============================
P("Study Guide — PDL Challenge 1", "title")
P("Neural Network-Based Function Upscaling: The Role of Activation Functions", "subtitle")
SP(4)
story.append(HRFlowable(width="100%", thickness=1.2, color=NAVY)); SP(10)

box("How to use this guide",
    "You are <b>not</b> expected to solve the challenge before the meeting — only to understand it. "
    "This guide gives you three things: (1) the minimum theory needed to read every sentence of the "
    "challenge PDF, (2) visual intuition for what the experiments will show (every figure here was "
    "produced with the challenge's own setup: N = 512, band-limit 50, seed 42), and (3) the vocabulary "
    "and questions to discuss it confidently.")

P("The challenge in one paragraph", "h1")
P("Generate smooth random functions (“band-limited”), keep only a few of their values (“samples”), "
  "and train neural networks to fill in the rest (“upscaling” / reconstruction). Classical signal "
  "processing solves this with linear formulas when enough samples exist. The challenge asks: what do "
  "<i>neural networks</i> do on this problem, and how does the answer depend on the <b>activation "
  "function</b>, the <b>architecture</b> (depth/width), and the <b>dataset size</b>?")

P("There are two tasks, and they test different things:", "body")
B("<b>Task 1 — fit one function.</b> A net with 1 input (position x) and 1 output (value y) memorises a "
  "single fixed function from M points. “Generalisation” = what the net draws <i>between</i> its "
  "training points.")
B("<b>Task 2 — learn the reconstruction operator.</b> A net with 100 inputs (the samples) and 512 outputs "
  "(the whole function) is trained on <i>many</i> random functions. “Generalisation” = reconstructing "
  "functions it has never seen.")
SP(2)

P("Reading map", "h2")
table(["Challenge PDF section", "What you need to know first", "Covered in"],
      [["Abstract + §1 Problem setting", "Fourier transform, band-limited functions, conjugate symmetry", "§1.1–1.3"],
       ["§1.1 sample code", "what each line does + one bug to fix", "§1.4"],
       ["§3 Architecture &amp; constraints", "neurons vs parameters, MSE loss, Adam optimiser", "§2.1–2.2"],
       ["§3.3 activation functions", "the activation gallery and why the choice dominates", "§2.3–2.4"],
       ["§4 Task 1 / §4.1 Task 2", "experiment playbooks, baselines, what to expect", "§3–4"],
       ["§5 Observe, observe, observe", "hypothesis-driven experimentation", "§5"]],
      [5.6 * cm, 7.6 * cm, 4.0 * cm])
story.append(PageBreak())

# ============================ SECTION 1 ============================
P("1&nbsp;&nbsp;The signal-processing half", "h1")
P("1.1&nbsp;&nbsp;A function as a recipe of sinusoids", "h2")
P("The Discrete Fourier Transform (DFT) rewrites any 512-point function as a sum of sinusoids. "
  "Frequency “bin” k holds a complex coefficient F[k] whose magnitude is the <b>amplitude</b> and whose "
  "angle is the <b>phase</b> of a sinusoid completing k cycles across the 512 samples. "
  "<b>Band-limited</b> simply means: all coefficients above some frequency are zero. The challenge "
  "generates functions with non-zero coefficients only for k = 1…49 — so nothing in the function "
  "oscillates faster than 49 cycles per 512 samples, which is why it looks smooth.")
fig(FIG + "/fig1_bandlimited.png",
    "Fig 1 — The actual challenge function (seed 42). Left: time domain. Right: its spectrum — energy only in "
    "bins 1…49 and their mirror images; every other bin is exactly zero.")

P("1.2&nbsp;&nbsp;Real functions ⇒ conjugate symmetry", "h2")
P("The generated function must be real-valued, and that forces the symmetry the PDF describes: "
  "F[−k] = F[k]* — magnitudes mirror (even symmetry), phases flip sign (odd symmetry). The code enforces "
  "exactly this when it copies the reversed conjugate coefficients into the negative-frequency slots.", "body")
fig(FIG + "/fig2_symmetry.png",
    "Fig 2 — Conjugate symmetry of the coefficients: magnitude is a mirror image around k = 0; phase is "
    "antisymmetric. This is what makes the inverse FFT come out real.")
box("Key number: 98",
    "Because of the symmetry, only the 49 coefficients k = 1…49 are free; each is one complex number = 2 real "
    "numbers. The DC term (k = 0) is set to zero. So <b>every function in this challenge is fully described by "
    "98 real numbers</b>. Remember 98 — it explains half of the experimental results below.",
    bg=AMBER, edge=AMBER_EDGE)

P("1.3&nbsp;&nbsp;Sampling: when is reconstruction even possible?", "h2")
P("The Nyquist–Shannon theorem, in this discrete setting: a function with highest frequency 49 is pinned down "
  "exactly by any set of samples rich enough to separate 98 unknowns — e.g. ≥ 98 well-placed points (uniform "
  "spacing needs the sampling rate to exceed 2×49 cycles). With enough samples, reconstruction is pure linear "
  "algebra (classically: sinc interpolation). With fewer, high frequencies masquerade as low ones — "
  "<b>aliasing</b> — and the information is genuinely gone: no method, neural or not, can recover it.", "body")
fig(FIG + "/fig3_nyquist.png",
    "Fig 3 — Linear least-squares reconstruction on the known sinusoid basis. Left: 128 regular samples (> 98) "
    "recover the function to machine precision. Right: 64 samples (< 98) — the reconstruction aliases; error is "
    "unavoidable no matter the method.")

P("1.4&nbsp;&nbsp;The generator code, decoded", "h2")
P("Line by line, the sample code: fix the seed (reproducibility) → draw random complex coefficients for bins "
  "1…49 → mirror them conjugated into the negative bins (forces a real signal) → inverse FFT → take "
  "<font face='DVM' size='8'>.real</font> (kills tiny numerical imaginary residue) → plot.", "body")
box("Two practical gotchas",
    "(1) <font face='DVM' size='8'>np.complex</font> was removed in NumPy ≥ 1.24 — write "
    "<font face='DVM' size='8'>dtype=complex</font> instead. "
    "(2) The generated values are small (≈ ±0.07, since ifft divides by N). Standardise y — and rescale "
    "x from 1…512 into [−1, 1] — before training, or optimisation will be needlessly slow.",
    bg=AMBER, edge=AMBER_EDGE)
story.append(PageBreak())

# ============================ SECTION 2 ============================
P("2&nbsp;&nbsp;The deep-learning half", "h1")
P("2.1&nbsp;&nbsp;MLPs, and what the “neuron budget” means", "h2")
P("Both tasks use plain fully-connected networks (MLPs): alternating linear layers and element-wise "
  "activations. The budget counts <b>hidden neurons (units), not parameters (weights)</b>. A 3×128 network "
  "uses 384 neurons — far under Task 1's 10,000 — but has ≈ 33,000 parameters "
  "(each layer contributes (n<sub>in</sub>+1)·n<sub>out</sub> weights). Task 2's million-neuron budget is generous: "
  "2×2048 uses only 4,096 neurons, yet already ≈ 5.5 million parameters. Depth vs width at a fixed budget "
  "is precisely one of the experiments.", "body")
fig(FIG + "/fig5_architectures.png",
    "Fig 4 — The two tasks as diagrams. Task 1: scalar-to-scalar curve fitting on one function. "
    "Task 2: vector-to-vector operator learning across many functions.")

P("2.2&nbsp;&nbsp;Training: MSE + Adam until convergence", "h2")
P("Both tasks minimise the mean squared error, MSE = (1/M) Σ (net(x<sub>i</sub>) − y<sub>i</sub>)², with the Adam "
  "optimiser (gradient descent with per-parameter adaptive step sizes — the robust default). "
  "“Until convergence” means: train until the loss curve flattens, not a fixed small number of epochs.", "body")
box("Practical training tips (they will save your experiments)",
    "• Normalise x to [−1, 1]; standardise y to zero mean, unit variance.<br/>"
    "• Task 1 datasets are tiny — use full-batch gradients; expect 10,000+ iterations.<br/>"
    "• Learning rate ≈ 10<super>−3</super> (sine/SIREN nets prefer ≈ 10<super>−4</super>); decay it near the end.<br/>"
    "• Always repeat with ≥ 3 random seeds — single runs can mislead.<br/>"
    "• Log train MSE <i>and</i> full-grid MSE separately; the gap is the interesting quantity.")

P("2.3&nbsp;&nbsp;The activation function gallery", "h2")
P("The activation is the only non-linear ingredient — without it a deep stack collapses into one linear map. "
  "Its shape (smoothness, saturation, periodicity) becomes the “texture” of everything the network draws.", "body")
fig(FIG + "/fig4_activations.png",
    "Fig 5 — Six activations and their derivatives. Solid: f(x); dashed: f′(x). The derivative determines both "
    "gradient flow during training and the smoothness of the fitted curve.")
table(["Activation", "Smooth?", "Saturates?", "Known issues", "When it shines"],
      [["ReLU", "no (corner at 0)", "no (for x&gt;0)", "dead neurons; piecewise-linear output", "deep nets, classification, speed"],
       ["Leaky ReLU", "no", "no", "still piecewise-linear", "like ReLU, avoids dead units"],
       ["Sigmoid", "yes", "both sides", "vanishing gradients; not zero-centred", "output layers for probabilities"],
       ["Tanh", "yes", "both sides", "vanishing gradients when deep", "smooth low-frequency regression"],
       ["GELU", "yes", "left side", "slightly costlier", "transformers; smooth ReLU substitute"],
       ["sin (SIREN)", "yes", "never", "needs special init + smaller lr", "signals, images, implicit fields — this challenge"]],
      [2.7 * cm, 2.2 * cm, 2.3 * cm, 4.9 * cm, 5.1 * cm])

P("2.4&nbsp;&nbsp;Why the activation dominates this particular problem", "h2")
P("Three ideas explain most of what you will observe:", "body")
B("<b>A ReLU network is a piecewise-linear function.</b> Compositions of linear maps and max(0,·) can only "
  "produce straight-line segments. Fitting a wave means approximating curves with many tiny segments, and "
  "between training points the net draws — literally — straight lines.")
B("<b>Spectral bias.</b> Gradient descent on standard MLPs fits low frequencies first and high frequencies "
  "reluctantly or never. This target has energy up to k = 49, so ReLU/tanh nets capture the slow envelope and "
  "miss the fast wiggles (watch the band edge in Fig 7).")
B("<b>Periodic activations build sinusoids natively.</b> A sine-activated net (SIREN) starts from the "
  "“correct basis” for band-limited signals — matching the model family to the signal class.")
fig(FIG + "/fig7_activation_fits.png",
    "Fig 6 — The core experiment of the whole challenge, run for you once: same 100 points, same 1→128→128→128→1 "
    "architecture, Adam until convergence; only the activation changes. ReLU (top) cannot even fit the training "
    "points (train MSE 0.13) and draws straight lines between them. Tanh (middle) memorises the points "
    "(2.6×10⁻³) but over-smooths between them. Sine (bottom) memorises to 10⁻³⁰ and interpolates best.")
fig(FIG + "/fig8_spectral_bias.png",
    "Fig 7 — The same three fits viewed in the frequency domain. All capture low frequencies; ReLU loses the "
    "most near the band edge (k ≈ 30–49); the sine net tracks the spectrum furthest. This is spectral bias made visible.")
box("The punchline (worth saying at the meeting)",
    "On the <i>same</i> 100 points, band-limit-aware least squares (§1.3) reconstructs the function <b>exactly</b> "
    "(MSE ≈ 10<super>−18</super>), while the best neural net reaches ≈ 0.47. Between data points, what a model draws is decided by "
    "its <b>inductive bias</b> — what its architecture and activation “naturally do” — not by the data, which is "
    "identical in all cases. The challenge is engineered so you discover this.",
    bg=AMBER, edge=AMBER_EDGE)
story.append(PageBreak())

# ============================ SECTION 3 ============================
P("3&nbsp;&nbsp;Task 1 playbook — memorise one function", "h1")
P("Fix one random band-limited function. Dataset = M of its 512 (x, y) pairs. Train, then evaluate on "
  "<b>all 512 grid positions</b>. Two numbers per run: train MSE (did it memorise?) and full-grid MSE "
  "(what did it draw in between?). Their gap is the interpolation quality — Task 1's real subject.", "body")
P("Effect of dataset size M", "h2")
P("The information threshold from §1 bites here: below 98 samples <i>no</i> method can identify the function, "
  "so the net fills gaps with its own prior — tanh draws calm flat curves, ReLU draws polylines. Above the "
  "threshold, fits snap into place rapidly:", "body")
fig(FIG + "/fig9_dataset_size.png",
    "Fig 8 — Tanh net, identical architecture and training, only M changes. M = 16: full-grid MSE 2.5 — worse than "
    "predicting the mean (!) despite train MSE 3×10⁻⁴. M = 64: envelope emerges (0.93). M = 256: nearly perfect (0.02). "
    "Note how the improvement is not gradual — it accelerates once M crosses the ≈ 98-sample information threshold.")
im_sp = PILImage.open(FIG + "/fig6_sampling_patterns.png")
story.append(KeepTogether([
    Paragraph("Effect of where the samples sit (the set Χ)", S["h2"]),
    Image(FIG + "/fig6_sampling_patterns.png", width=FRAME_W,
          height=FRAME_W * im_sp.size[1] / im_sp.size[0]),
    Paragraph("Fig 9 — Three choices of 100 sample positions. Regular spacing: largest gap 6. Random: largest gap 35. "
              "Exponential: rounding to integers collapses 100 requested points into 71 unique ones, with 31-sample deserts on "
              "the right — count your unique indexes! Expect the local error to track the local gap size.", S["caption"])]))
P("Suggested experiment grid", "h2")
table(["Vary", "Values to try", "Hold fixed", "Plot"],
      [["dataset size M", "10, 25, 50, 100, 200, 400", "3×128 net, activation, random Χ, 3 seeds", "full-grid MSE vs M (log–log) — look for the elbow near 98"],
       ["activation", "relu, tanh, sine (+ gelu)", "M = 100, architecture, Χ", "overlay fits (as Fig 6); bar chart of MSEs"],
       ["depth × width (equal budget)", "1×384, 3×128, 6×64", "M, activation, Χ", "MSE vs depth; note trainability of deep tanh"],
       ["sampling pattern Χ", "regular / random / exponential", "M = 100, net", "error vs position, with sample locations marked"]],
      [3.3 * cm, 4.6 * cm, 4.9 * cm, 4.4 * cm])
P("What you should expect (hypotheses to verify, not trust)", "h2")
B("ReLU shows the largest full-grid error and visible straight segments; sine the smallest; tanh in between.")
B("Full-grid MSE vs M drops steeply around M ≈ 100 and then saturates at a floor set by the activation.")
B("At equal neuron budget, moderate depth beats both extremes: 1 wide layer underfits curvature; very deep tanh gets hard to optimise.")
B("Errors concentrate in sampling gaps; exponential spacing fails on the sparse side even for large M.")
story.append(PageBreak())

# ============================ SECTION 4 ============================
P("4&nbsp;&nbsp;Task 2 playbook — learn the reconstruction operator", "h1")
P("Now the network never sees x at all. Input: the vector of 100 sampled values f(Χ); output: all 512 values "
  "of f. Training set: M different random functions. The net is learning the <b>operator</b> “samples → "
  "function”, i.e. exactly what sinc interpolation does in classical DSP. Crucially, test it on <b>new</b> "
  "random functions the net never saw — this is true generalisation across a function class, unlike Task 1's "
  "within-one-function interpolation.", "body")
box("The insight that reframes the whole task",
    "Reconstruction from samples is a <b>linear</b> map (§1.3): each output value is a fixed weighted sum of the "
    "100 inputs. With regular Χ, 100 samples ≥ 98 degrees of freedom, so <b>a single linear layer with no "
    "activation can already be essentially exact</b>. Run that baseline first. The scientific question then "
    "becomes: do nonlinear activations help, hurt, or merely need more data to match a linear model — and does "
    "the answer flip when Χ is irregular (ill-conditioned) or samples are too few?",
    bg=AMBER, edge=AMBER_EDGE)
P("What changes compared to Task 1", "h2")
B("<b>Generalisation axis:</b> across functions, not across positions. Always keep a validation set of fresh functions (different seed).")
B("<b>Dataset size M now counts functions.</b> The input vectors span a 98-dimensional subspace, so interesting "
  "behaviour starts near M ≈ 100; sweep M ∈ {50, 100, 200, 500, 2000, 10000} and find the plateau.")
B("<b>Architecture:</b> the budget (10⁶ neurons) is not the binding constraint — parameters and memory are. "
  "2×2048 ReLU (≈ 5.5 M parameters) is a solid starting point; mini-batches of 128–512 with Adam.")
B("<b>Choice of Χ:</b> with irregular or exponential spacing the linear problem becomes ill-conditioned "
  "(and with the naive exponential grid, only 71 unique indexes — fewer than 98 — so even perfect linear "
  "reconstruction is impossible). Plot per-position validation error with sample locations marked: errors pool in the gaps.")
P("Suggested experiment grid", "h2")
table(["Vary", "Values to try", "Hold fixed", "Plot"],
      [["model class", "linear (no activation), relu, tanh", "M = 2000, regular Χ", "val MSE bar chart — can anything beat linear?"],
       ["dataset size M", "50 … 10000", "2×2048 relu, regular Χ", "val MSE vs M (log–log): decay then plateau"],
       ["sampling set Χ", "regular / random / exponential", "M = 2000, model", "per-position error vs sample locations"],
       ["depth × width", "1×4096, 2×2048, 4×1024", "M, Χ", "val MSE + training time"]],
      [3.3 * cm, 4.6 * cm, 4.9 * cm, 4.4 * cm])
P("What you should expect", "h2")
B("With regular Χ the linear model is near-exact; nonlinear nets approach it given enough M but rarely beat it on clean data.")
B("Validation error falls roughly as a power law in M, then plateaus once the operator is identified.")
B("Irregular Χ: errors grow inside large gaps; nonlinearity + more data help somewhat, but conditioning rules.")
B("If you add noise to the samples (optional experiment), nonlinear nets can start to win — priors matter once information is imperfect.")
story.append(PageBreak())

# ============================ SECTION 5 ============================
P("5&nbsp;&nbsp;Run experiments like a scientist (challenge §5)", "h1")
P("The challenge's final section is really the grading rubric: it wants hypothesis-driven experiments, not a "
  "pile of loss curves. For each claim: define variables, control confounders, span a wide range of conditions, "
  "replicate over seeds, and only then conclude. Four ready-to-test hypotheses:", "body")
table(["Hypothesis", "Independent variable", "Control (hold fixed)", "Evidence to collect"],
      [["H1: full-grid error decays as a power law in M until an information/capacity floor",
        "M", "architecture, activation, Χ, seeds", "log–log MSE vs M; slope + elbow location (≈ 98?)"],
       ["H2: smooth/periodic activations beat ReLU on band-limited targets at equal budget",
        "activation", "M, architecture, Χ", "MSE table + fitted curves + spectra (Figs 6–7)"],
       ["H3 (Task 2): a linear model matches nonlinear nets when Χ is regular and clean",
        "model class", "M, Χ", "val MSE linear vs relu vs tanh across M"],
       ["H4: local error is predicted by local sampling gap, not by M alone",
        "Χ pattern", "M, model", "error-vs-position overlaid with gap sizes"]],
      [5.4 * cm, 3.0 * cm, 3.9 * cm, 4.9 * cm])
box("Experiment hygiene checklist",
    "• Fix and record every seed (data, init, batch order). • Same convergence criterion for every run. "
    "• ≥ 3 seeds per configuration; report mean ± std. • Change one variable at a time. "
    "• Save loss curves, not just final numbers. • Normalise identically across runs — otherwise MSEs are not comparable.")

# ============================ SECTION 6 ============================
P("6&nbsp;&nbsp;Self-test before the meeting", "h1")
qa = [
    ("Why are the Fourier coefficients conjugate-symmetric?",
     "Because the time-domain function is real-valued; realness ⇔ F[−k] = F[k]*."),
    ("How many real numbers fully describe one generated function?",
     "98 (49 free complex coefficients × 2; DC term is zero)."),
    ("Roughly how many samples does perfect classical reconstruction need?",
     "≥ 98 well-placed samples (uniform sampling above the Nyquist rate for kₘₐₓ = 49). Below that: aliasing, information lost."),
    ("Why is a ReLU network's output piecewise linear?",
     "It composes linear maps with max(0,·); every region of input space gets one affine formula."),
    ("What is spectral bias?",
     "Gradient-trained MLPs learn low frequencies first; high-frequency detail comes late or never."),
    ("How is generalisation measured differently in the two tasks?",
     "Task 1: unseen positions of the same function (full-grid MSE). Task 2: entirely unseen functions (validation set)."),
    ("Why could a purely linear network solve Task 2 with regular Χ?",
     "Reconstruction from ≥ 98 informative samples is a linear operator — 100 regular samples suffice."),
    ("Why repeat runs with several seeds?",
     "Initialisation and data draws add variance; conclusions need mean ± std, not one lucky run."),
]
for q, a in qa:
    story.append(KeepTogether([Paragraph("Q. " + q, S["qa_q"]), Paragraph("A. " + a, S["qa_a"])]))
SP(8)
P("Mini-glossary", "h2")
P("<b>DFT bin</b>: one frequency slot of the discrete Fourier transform. <b>Band-limit</b>: highest non-zero "
  "frequency. <b>Nyquist rate</b>: minimum sampling rate (2× highest frequency) for exact linear recovery. "
  "<b>Aliasing</b>: high frequencies disguising as low ones after undersampling. <b>Sinc interpolation</b>: the "
  "classical linear reconstruction formula. <b>Neuron vs parameter</b>: hidden unit vs trainable weight. "
  "<b>Epoch/iteration</b>: pass over data / single gradient step. <b>Adam</b>: adaptive-step gradient "
  "optimiser. <b>Overfitting</b>: excellent on training data, poor elsewhere — <i>deliberate</i> in Task 1. "
  "<b>Generalisation gap</b>: train-vs-test error difference. <b>Inductive bias</b>: what a model family does "
  "where data is silent. <b>Spectral bias</b>: see self-test. <b>SIREN</b>: MLP with sine activations and "
  "matched initialisation (Sitzmann et al., 2020).", "body")
P("Good questions to ask at the meeting", "h2")
B("Is the neuron budget counting hidden units only, and are there constraints on parameters or training time?")
B("For Task 1, should the evaluation grid be exactly the 512 integer positions, or also off-grid x?")
B("What convergence criterion should we standardise on, so results are comparable between students?")
B("May we add a noisy-samples variant in Task 2 to test whether nonlinearity starts to pay off?")
story.append(PageBreak())

# ============================ APPENDICES ============================
P("Appendix A — Task 1 starter code (task1_starter.py)", "h1")
P("Also provided as a separate runnable file. PyTorch; sweeps activations at M = 100; TODOs mark the "
  "remaining experiment axes.", "body")
code1 = open(OUT + "/task1_starter.py").read()
story.append(Preformatted(code1, S["code"]))
story.append(PageBreak())
P("Appendix B — Task 2 starter code (task2_starter.py)", "h1")
P("Trains the operator net on M random functions and validates on unseen ones; the linear baseline is one "
  "argument away (act=“linear”).", "body")
code2 = open(OUT + "/task2_starter.py").read()
story.append(Preformatted(code2, S["code"]))

# ============================ BUILD ============================
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DVS", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(MARGIN, 1.05 * cm, "PDL Challenge 1 — Study Guide")
    canvas.drawRightString(PAGE_W - MARGIN, 1.05 * cm, f"page {doc.page}")
    canvas.restoreState()

doc = BaseDocTemplate(OUT + "/PDL_Challenge_1_Study_Guide.pdf", pagesize=A4,
                      leftMargin=MARGIN, rightMargin=MARGIN,
                      topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                      title="PDL Challenge 1 — Study Guide",
                      author="prepared for Toodmuk")
frame = Frame(MARGIN, 1.6 * cm, FRAME_W, PAGE_H - 1.5 * cm - 1.6 * cm, id="main")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(story)
print("PDF built")
