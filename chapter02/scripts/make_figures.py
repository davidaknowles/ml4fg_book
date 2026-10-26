"""Generate the computed figures and tables for the CNN and gene regulation chapter.

Each function writes one PDF (or a LaTeX table fragment) to ../figures/. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from contourpy import contour_generator
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from plotnine import (
    aes, annotate, coord_cartesian, element_blank, element_text, facet_wrap,
    geom_hline, geom_line, geom_path, geom_point, geom_polygon, geom_vline, ggplot,
    labs, scale_color_manual, scale_fill_identity, scale_linetype_manual,
    scale_shape_manual, scale_x_continuous, scale_y_continuous, theme, theme_bw,
)
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
PAL = ["#1b6ca8", "#d1495b", "#66a182", "#edae49", "#3d3d3d"]
NUC = "ACGT"
NUC_COL = {"A": "#2e8b3a", "C": "#1f4fd1", "G": "#f0a020", "T": "#d62728"}

THEME = theme_bw(base_size=10) + theme(
    panel_grid_minor=element_blank(),
    strip_background=element_blank(),
    strip_text=element_text(size=10),
    legend_title=element_blank(),
)


def save(p, name, w=5.0, h=3.2):
    p.save(OUT / name, width=w, height=h, verbose=False, metadata={"CreationDate": None})


def contour_paths(xx, yy, zz, levels):
    """Contour lines of zz on the grid as a long data frame (level, piece, x, y)."""
    gen = contour_generator(xx, yy, zz)
    rows = []
    for lev in levels:
        for k, seg in enumerate(gen.lines(lev)):
            rows.append(pd.DataFrame({"x": seg[:, 0], "y": seg[:, 1],
                                      "level": lev, "piece": f"{lev}_{k}"}))
    return pd.concat(rows, ignore_index=True)


def one_hot(seq):
    """4 x L one-hot encoding with rows A, C, G, T."""
    X = np.zeros((4, len(seq)))
    for i, c in enumerate(seq):
        X[NUC.index(c), i] = 1.0
    return X


def conv1d(X, W, pad_left=0, pad_right=0, stride=1):
    """Cross-correlation of a C_in x L input with a C_out x C_in x K filter bank (no bias)."""
    Xp = np.pad(X, ((0, 0), (pad_left, pad_right)))
    K = W.shape[-1]
    L_out = (Xp.shape[1] - K) // stride + 1
    return np.stack([[np.sum(W[d] * Xp[:, j * stride:j * stride + K]) for j in range(L_out)]
                     for d in range(W.shape[0])])


def maxpool1d(Z, width, stride):
    L_out = (Z.shape[1] - width) // stride + 1
    return np.stack([Z[:, j * stride:j * stride + width].max(1) for j in range(L_out)], 1)


relu = lambda z: np.maximum(z, 0.0)

# GATA PWM used in the slides (rows A, C, G, T; 6 positions), in log-odds units.
W_GATA = np.array([[-5.7, -3.2, 3.7, -3.2, 3.7, 0.6],
                   [0.5, -3.2, -3.2, -3.2, -3.2, -5.7],
                   [0.5, 3.7, -3.2, -3.2, -3.2, -5.7],
                   [-5.7, -3.2, -3.2, 3.7, -3.2, 0.5]])
# Toy NANOG-like core motif (C/G)ATTA(C/G).
W_NANOG = np.array([[-3.2, 3.7, -3.2, -3.2, 3.7, -1.2],
                    [2.0, -3.2, -3.2, -3.2, -3.2, 1.2],
                    [1.5, -3.2, -3.2, -3.2, -3.2, 1.3],
                    [-3.2, -3.2, 3.7, 3.7, -3.2, -1.4]])
# Toy sequences bound by a TF (the Wikipedia position weight matrix example).
WIKI_SEQS = ["GAGGTAAAC", "TCCGTAAGT", "CAGGTTGGA", "ACAGTCAGT", "TAGGTCATT",
             "TAGGTACTG", "ATGGTAACT", "CAGGTATAC", "TGTGTGAGT", "AAGGTAAGT"]


# ---------------------------------------------------------------- neural networks

def fig_limitations():
    """(a) A nonlinear 1D relationship; (b) a class region that no linear boundary separates."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, 80)
    f = lambda t: 0.25 * t ** 3 - 1.2 * t + 0.6 * np.sin(2 * t)
    y = f(x) + rng.normal(0, 0.4, x.size)
    xg = np.linspace(-3, 3, 300)
    lin = LinearRegression().fit(x[:, None], y)
    mlp = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=20000, random_state=0,
                       learning_rate_init=0.01, alpha=1e-3).fit(x[:, None], y)
    fits = pd.concat([
        pd.DataFrame({"x": xg, "y": lin.predict(xg[:, None]), "fit": "linear regression"}),
        pd.DataFrame({"x": xg, "y": mlp.predict(xg[:, None]), "fit": "neural network (MLP)"}),
    ])
    p = (ggplot(pd.DataFrame({"x": x, "y": y}), aes("x", "y")) + geom_point(size=1.2, color="grey")
         + geom_line(fits, aes(color="fit", linetype="fit"), size=0.9)
         + scale_color_manual(values=[PAL[1], PAL[0]]) + scale_linetype_manual(values=["solid", "solid"])
         + labs(x="$x$", y="$y$") + THEME + theme(legend_position="top"))
    save(p, "limitations_nonlinear.pdf", 3.4, 3.2)

    n = 60
    X = rng.uniform(0, 3, (n, 2))
    yb = (X[:, 0] * X[:, 1] > 1).astype(int)
    lr = LogisticRegression(C=100).fit(X, yb)
    nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation="tanh", max_iter=20000, alpha=1e-3, random_state=0).fit(X, yb)
    g = np.linspace(0, 3, 250)
    g1, g2 = np.meshgrid(g, g)
    G = np.c_[g1.ravel(), g2.ravel()]
    bounds = []
    for name, m in [("logistic regression", lr), ("neural network (MLP)", nn)]:
        c = contour_paths(g1, g2, m.predict_proba(G)[:, 1].reshape(g1.shape), [0.5])
        c["fit"] = name
        bounds.append(c)
    b = pd.concat(bounds)
    b["piece"] = b["fit"] + b["piece"]
    pts = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "cls": np.where(yb == 1, "$+$", "$-$")})
    p = (ggplot() + geom_point(pts, aes("x1", "x2", shape="cls"), size=4, stroke=1, color="#555555")
         + geom_path(b, aes("x", "y", group="piece", color="fit"), size=0.9)
         + scale_shape_manual(values={"$+$": "+", "$-$": "_"}, guide=None)
         + scale_color_manual(values=[PAL[1], PAL[0]])
         + coord_cartesian(xlim=(0, 3), ylim=(0, 3))
         + labs(x="$x_1$", y="$x_2$") + THEME + theme(legend_position="top"))
    save(p, "limitations_interaction.pdf", 3.4, 3.4)


def fig_activations():
    z = np.linspace(-4, 4, 400)
    acts = {"logistic sigmoid $\\sigma(z)$": expit(z), "ReLU $\\max(z,0)$": relu(z),
            "GELU $z\\,\\Phi(z)$": z * norm.cdf(z)}
    df = pd.concat([pd.DataFrame({"z": z, "a": v, "panel": k}) for k, v in acts.items()])
    df["panel"] = pd.Categorical(df["panel"], list(acts))
    p = (ggplot(df, aes("z", "a")) + geom_line(color=PAL[0], size=0.8)
         + geom_hline(yintercept=0, color="grey") + geom_vline(xintercept=0, color="grey")
         + facet_wrap("~panel", scales="free_y") + labs(x="$z$", y="activation") + THEME)
    save(p, "activations.pdf", 6.0, 2.3)


def random_mlp(rng, n_layers, width=5):
    Ws = [rng.normal(0, 1, (1 if m == 0 else width, width)) for m in range(n_layers)]
    bs = [rng.normal(0, 1, width) for _ in range(n_layers)]
    w_out = rng.normal(0, 1, width)

    def f(x):
        h = x[:, None]
        for W, b in zip(Ws, bs):
            h = relu(h @ W + b)
        return h @ w_out
    return f


def fig_mlp_random():
    """Functions computed by randomly initialized ReLU MLPs with 1-4 hidden layers of 5 units."""
    rng = np.random.default_rng(11)
    x = np.linspace(-5, 5, 2001)
    rows = []
    for M in [1, 2, 3, 4]:
        for r in range(3):
            y = random_mlp(rng, M)(x)
            y = (y - y.mean()) / (y.std() + 1e-12)
            rows.append(pd.DataFrame({"x": x, "y": y, "draw": str(r), "M": f"$M={M}$ hidden layer" + ("s" if M > 1 else "")}))
    df = pd.concat(rows)
    df["M"] = pd.Categorical(df["M"], df["M"].unique())
    p = (ggplot(df, aes("x", "y", color="draw")) + geom_line(size=0.6)
         + facet_wrap("~M", nrow=1) + scale_color_manual(values=PAL[:3], guide=None)
         + labs(x="input $x$", y="output (standardized)") + THEME)
    save(p, "mlp_random.pdf", 6.5, 2.3)


def fig_sgd():
    """Full-batch gradient descent, SGD and Adam on a two-parameter logistic regression loss."""
    rng = np.random.default_rng(3)
    N = 1000
    X = rng.normal(0, 1, (N, 2)) @ np.array([[1.0, 0.0], [0.6, 0.8]])
    y = (rng.random(N) < expit(X @ np.array([2.0, -1.0]))).astype(float)

    def loss(B):  # B: n x 2
        G = X @ B.T
        return np.mean(np.logaddexp(0, G) - y[:, None] * G, 0)

    def grad(b, idx):
        return X[idx].T @ (expit(X[idx] @ b) - y[idx]) / len(idx)

    start, T, bs = np.array([-1.5, 1.5]), 150, 10
    paths = []
    b = start.copy(); tr = [b.copy()]
    for t in range(T):
        b = b - 1.0 * grad(b, np.arange(N)); tr.append(b.copy())
    paths.append(("gradient descent, $\\alpha=1$", np.array(tr)))
    b = start.copy(); tr = [b.copy()]
    for t in range(1, T + 1):
        idx = rng.choice(N, bs, replace=False)
        b = b - 5.0 / (5.0 + t) * grad(b, idx); tr.append(b.copy())
    paths.append(("SGD, $|B|=10$, $\\alpha_t=5/(5+t)$", np.array(tr)))
    b = start.copy(); tr = [b.copy()]
    m = v = np.zeros(2); b1, b2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    for t in range(1, T + 1):
        idx = rng.choice(N, bs, replace=False)
        g = grad(b, idx)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        b = b - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps); tr.append(b.copy())
    paths.append(("Adam, $|B|=10$, $\\alpha=0.1$", np.array(tr)))
    df = pd.concat([pd.DataFrame({"x": P[:, 0], "y": P[:, 1], "method": k}) for k, P in paths])
    df["method"] = pd.Categorical(df["method"], [k for k, _ in paths])
    g1, g2 = np.meshgrid(np.linspace(-2, 3.5, 200), np.linspace(-2.5, 2.5, 200))
    zz = loss(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    c = contour_paths(g1, g2, zz, zz.min() + np.array([0.01, 0.05, 0.15, 0.3, 0.5, 0.8]))
    bhat = np.array([g1.ravel()[zz.argmin()], g2.ravel()[zz.argmin()]])
    p = (ggplot() + geom_path(c, aes("x", "y", group="piece"), color="grey", size=0.4)
         + geom_path(df, aes("x", "y", color="method"), size=0.6)
         + geom_point(pd.DataFrame({"x": [start[0]], "y": [start[1]]}), aes("x", "y"), size=2)
         + annotate("point", x=bhat[0], y=bhat[1], shape="x", size=3)
         + facet_wrap("~method") + scale_color_manual(values=PAL[:3], guide=None)
         + coord_cartesian(xlim=(-2, 3.5), ylim=(-2.5, 2.5))
         + labs(x="$\\beta_1$", y="$\\beta_2$") + THEME)
    save(p, "sgd_paths.pdf", 6.5, 2.6)


# ---------------------------------------------------------------- sequence logos

_FONT = FontProperties(family="DejaVu Sans", weight="bold")
_GLYPHS = {}


def glyph(letter):
    """Polygons of a letter normalized to the unit square, as (outer, holes)."""
    if letter not in _GLYPHS:
        tp = TextPath((0, 0), letter, size=1, prop=_FONT)
        polys = [np.asarray(q) for q in tp.to_polygons()]
        allp = np.vstack(polys)
        lo, hi = allp.min(0), allp.max(0)
        polys = [(q - lo) / (hi - lo) for q in polys]
        area = lambda q: 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
        polys.sort(key=area, reverse=True)
        outer, rest = [polys[0]], []
        for q in polys[1:]:
            (rest if MplPath(polys[0]).contains_point(q.mean(0)) else outer).append(q)
        _GLYPHS[letter] = (outer, rest)
    return _GLYPHS[letter]


def logo_polygons(H):
    """Letter polygons for a 4 x K height matrix (rows A, C, G, T). Negative heights stack below zero."""
    rows, k = [], 0
    for j in range(H.shape[1]):
        for sign in (1, -1):
            hs = [(H[n, j], NUC[n]) for n in range(4) if sign * H[n, j] > 1e-6]
            # positive letters: smallest at the bottom; negative letters: smallest nearest zero
            hs.sort(key=lambda t: abs(t[0]))
            base = 0.0
            for h, letter in hs:
                y0, y1 = (base, base + h) if sign > 0 else (base - abs(h), base)
                base = y1 if sign > 0 else y0
                outer, holes = glyph(letter)
                for q, col in [(q, NUC_COL[letter]) for q in outer] + [(q, "white") for q in holes]:
                    rows.append(pd.DataFrame({"x": j + 1 - 0.45 + 0.9 * q[:, 0], "y": y0 + (y1 - y0) * q[:, 1],
                                              "g": k, "fill": col}))
                    k += 1
    return pd.concat(rows, ignore_index=True)


def pfm_ppm_pwm():
    C = np.array([[sum(s[i] == n for s in WIKI_SEQS) for i in range(len(WIKI_SEQS[0]))] for n in NUC], float)
    P = C / C.sum(0)
    with np.errstate(divide="ignore"):
        W = np.log2(P / 0.25)
    return C, P, W


def fig_logos():
    _, P, _ = pfm_ppm_pwm()
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = 2 + np.nansum(np.where(P > 0, P * np.log2(P), 0), 0)
    df = logo_polygons(P * ic)
    p = (ggplot(df, aes("x", "y", group="g", fill="fill")) + geom_polygon() + scale_fill_identity(guide=None)
         + scale_x_continuous(breaks=list(range(1, P.shape[1] + 1)))
         + scale_y_continuous(limits=(0, 2)) + labs(x="position $k$", y="information (bits)") + THEME)
    save(p, "logo_wiki.pdf", 3.4, 2.2)
    df = logo_polygons(W_GATA)
    p = (ggplot(df, aes("x", "y", group="g", fill="fill")) + geom_polygon() + scale_fill_identity(guide=None)
         + geom_hline(yintercept=0, color="grey")
         + scale_x_continuous(breaks=list(range(1, 7)))
         + labs(x="position $k$", y="weight $W_{n,k}$") + THEME)
    save(p, "logo_gata.pdf", 3.4, 2.6)


# ---------------------------------------------------------------- tables and worked examples

def fmt(v, nd=2):
    if np.isneginf(v):
        return "$-\\infty$"
    s = f"{v:.{nd}f}"
    return s.replace("-", "$-$")


def table_pwm_example():
    """PFM, PPM and PWM for the toy bound sequences, and the score of GAGGTAAAC."""
    C, P, W = pfm_ppm_pwm()
    lines = []
    for name, M, nd in [("counts $C_{n,k}$", C, 0), ("probabilities $P_{n,k}$", P, 1), ("weights $W_{n,k}$", W, 2)]:
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{name}}}}} \\\\")
        for n in range(4):
            lines.append(NUC[n] + " & " + " & ".join(fmt(v, nd) for v in M[n]) + " \\\\")
        lines.append("\\midrule")
    head = ["\\begin{tabular}{l*{9}{r}}", "\\toprule",
            "position $k$ & " + " & ".join(str(k) for k in range(1, 10)) + " \\\\", "\\midrule"]
    (OUT / "pwm_example_table.tex").write_text("\n".join(head + lines[:-1] + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    s = "GAGGTAAAC"
    pS = np.prod([P[NUC.index(c), i] for i, c in enumerate(s)])
    score = sum(W[NUC.index(c), i] for i, c in enumerate(s))
    odds = pS / 0.25 ** len(s)
    print(f"  p(x|b=1)={pS:.7f} odds={odds:.1f} log2odds={score:.3f} 2^score={2**score:.1f} P(b=1|x)={odds/(1+odds):.4f}")


def table_pwm_scan():
    """Scan the GATA PWM along a 13 nt sequence (padding 2 left, 3 right), then ReLU and max pooling."""
    seq = "GCATTACCGATAA"
    X = one_hot(seq)
    z = conv1d(X, W_GATA[None], pad_left=2, pad_right=3)[0]
    pooled = maxpool1d(z[None], 4, 4)[0]
    rows = ["sequence & " + " & ".join(seq) + " \\\\", "\\midrule"]
    for n in range(4):
        rows.append(f"{NUC[n]} & " + " & ".join(str(int(v)) for v in X[n]) + " \\\\")
    rows.append("\\midrule")
    rows.append("score $z_j$ & " + " & ".join(fmt(v, 1) for v in z) + " \\\\")
    rows.append("ReLU & " + " & ".join(fmt(v, 1) for v in relu(z)) + " \\\\")
    cells = []
    for b in range(3):
        cells.append(f"\\multicolumn{{4}}{{c}}{{{fmt(pooled[b], 1)}}}")
    rows.append("max pool & " + " & ".join(cells) + " & \\\\")
    head = ["\\begin{tabular}{l*{13}{r}}", "\\toprule"]
    (OUT / "pwm_scan_table.tex").write_text("\n".join(head + rows + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    print("  scan", np.round(z, 1), "pooled", np.round(pooled, 1))


def toy_cnn():
    """The two-filter toy CNN of the lecture, made deterministic. Prints every intermediate value."""
    rng = np.random.default_rng(0)
    seq = "CGCATTACGATAA"
    X = one_hot(seq)
    W1 = rng.normal(0, 1, (3, 4, 6)) - 2.0
    W1[0], W1[1] = W_GATA, W_NANOG
    H1 = relu(conv1d(X, W1))
    P1 = maxpool1d(H1, 2, 2)
    W2 = rng.normal(0, 1, (2, 3, 3)) - 2.0
    W2[0] = np.array([[-0.4, -0.2, 0.7], [0.6, -0.9, -1.2], [-1.1, -1.2, -1.0]])
    H2 = relu(conv1d(P1, W2))
    v = H2.ravel()
    W3 = rng.normal(0, 1, (3, 4)) - 2.0
    W3[0, 1] = 0.3
    H3 = relu(W3 @ v)
    W4 = rng.normal(0, 1, (1, 3)) - 2.0
    W4[0, 0] = 0.5
    logit = (W4 @ H3)[0]
    for name, val in [("conv1+relu", H1), ("pool", P1), ("conv2+relu", H2), ("dense1", H3)]:
        print(f"  {name}:\n{np.round(val, 4)}")
    print(f"  logit={logit:.4f} prob={expit(logit):.4f}")
    assert np.isclose(H1[0, 7], 15.9) and np.isclose(H1[1, 2], 18.0) and np.isclose(H2[0, 1], 19.77)
    assert np.isclose(logit, 2.9655) and np.allclose(H1[2], 0) and np.allclose(H3[1:], 0)


def check_backprop():
    """Finite-difference check of the one-hidden-layer MLP gradients derived in the text."""
    rng = np.random.default_rng(5)
    P_, H_, D_ = 4, 3, 2
    x, yv = rng.normal(size=P_), rng.normal(size=D_)
    B, A = rng.normal(size=(P_, H_)), rng.normal(size=(H_, D_))

    def lossf(B, A):
        o = expit(x @ B) @ A
        return 0.5 * np.sum((o - yv) ** 2)
    g = x @ B; h = expit(g); o = h @ A
    d_o = o - yv
    gA = np.outer(h, d_o)
    d_g = (A @ d_o) * h * (1 - h)
    gB = np.outer(x, d_g)
    eps = 1e-6
    num = np.zeros_like(B)
    for i in range(P_):
        for j in range(H_):
            E = np.zeros_like(B); E[i, j] = eps
            num[i, j] = (lossf(B + E, A) - lossf(B - E, A)) / (2 * eps)
    numA = np.zeros_like(A)
    for i in range(H_):
        for j in range(D_):
            E = np.zeros_like(A); E[i, j] = eps
            numA[i, j] = (lossf(B, A + E) - lossf(B, A - E)) / (2 * eps)
    assert np.allclose(num, gB, atol=1e-6) and np.allclose(numA, gA, atol=1e-6)
    print("  backprop gradients match finite differences")


def chipseq_poisson():
    from scipy.stats import poisson
    for lam, k in [(2.0, 15), (5.0, 15)]:
        print(f"  P(X>={k} | lambda={lam}) = {poisson.sf(k - 1, lam):.2e}")


if __name__ == "__main__":
    for fn in [fig_limitations, fig_activations, fig_mlp_random, fig_sgd, fig_logos,
               table_pwm_example, table_pwm_scan, toy_cnn, check_backprop, chipseq_poisson]:
        print(fn.__name__)
        fn()
