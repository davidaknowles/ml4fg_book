"""Generate the computed figures for the interpretation chapter and check the worked examples.

Each fig_ function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotnine import (
    aes, annotate, element_blank, element_text, facet_wrap, geom_area, geom_col,
    geom_hline, geom_line, geom_point, geom_rect, geom_segment, geom_text, geom_vline,
    ggplot, labs, scale_color_manual, scale_fill_manual, scale_linetype_manual,
    theme, theme_bw, xlim,
)
from scipy.special import expit
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"

from attribution import (SeqCNN, backprop_scores, deeplift, integrated_gradients, ism,
                         shapley)

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
PAL = ["#1b6ca8", "#d1495b", "#66a182", "#edae49", "#3d3d3d"]
BASE_COL = {"A": "#2e9e44", "C": "#1b6ca8", "G": "#edae49", "T": "#d1495b"}

THEME = theme_bw(base_size=10) + theme(
    panel_grid_minor=element_blank(),
    strip_background=element_blank(),
    strip_text=element_text(size=10),
    legend_title=element_blank(),
)


def save(p, name, w=5.0, h=3.2):
    p.save(OUT / name, width=w, height=h, verbose=False, metadata={"CreationDate": None})


relu = lambda z: np.maximum(z, 0)


# ------------------------------------------------------------ worked examples

def check_toy_examples():
    """Recompute every number quoted in the worked examples of the chapter."""
    # Toy CNN (gradients, deconvnet, guided backprop)
    W1 = np.array([[-2, 2, 0, 0, 0], [0, 1, -2, 0, 0], [0, 0, -1, 3, 0], [0, 0, 0, -3, 4]], float)
    W2 = np.array([[1, -1, -2, 0], [0, 1, 2, -1]], float)

    def backward(x, top, rule):
        z1 = W1 @ x; z2 = W2 @ relu(z1)
        g2 = top(relu(z2))
        g2 = {"grad": g2 * (z2 > 0), "deconv": relu(g2), "guided": relu(g2) * (z2 > 0)}[rule]
        g1 = W2.T @ g2
        g1 = {"grad": g1 * (z1 > 0), "deconv": relu(g1), "guided": relu(g1) * (z1 > 0)}[rule]
        return W1.T @ g1, z1, z2

    x = np.array([2, 1, 2, 1, 1], float)
    maxswitch = lambda a: (a == a.max()).astype(float)
    for rule, want in [("grad", [0, 0, -2, 9, -4]), ("deconv", [0, 1, -4, 6, 0]),
                       ("guided", [0, 0, -2, 6, 0])]:
        g, z1, z2 = backward(x, maxswitch, rule)
        assert np.allclose(g, want), (rule, g)
    assert np.allclose(z1, [-2, -3, 1, 1]) and np.allclose(z2, [-2, 1])
    # Exercise network: same second layer, linear output 2 a - b
    W1[:] = [[-1, 2, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, -1, 3, 0], [0, 0, 0, -3, 4]]
    x = np.array([0, 2, 1, 0, 3], float)
    top = lambda a: np.array([2.0, -1.0])
    for rule, want in [("grad", [-2, 2, -2, 0, 0]), ("deconv", [-2, 4, 0, 0, 0]),
                       ("guided", [-2, 4, 0, 0, 0])]:
        g, z1, z2 = backward(x, top, rule)
        assert np.allclose(g, want), (rule, g)
    assert np.allclose(z1, [4, 3, -1, 12]) and np.allclose(z2, [1, -9])

    # DeepLIFT Rescale on a one-hidden-layer net y = v . relu(W x + b) + c
    def dl(W, b, v, c, x, x0):
        z, z0 = W @ x + b, W @ x0 + b
        dz = z - z0
        m_relu = np.where(np.abs(dz) > 0, (relu(z) - relu(z0)) / np.where(dz == 0, 1, dz), z > 0)
        m = W.T @ (v * m_relu)
        y, y0 = v @ relu(z) + c, v @ relu(z0) + c
        return m * (x - x0), y - y0
    # saturation example
    C, d = dl(np.array([[-1.0, -1.0]]), np.array([1.0]), np.array([-1.0]), 1.0,
              np.array([1.0, 1.0]), np.zeros(2))
    assert np.allclose(C, [0.5, 0.5]) and np.isclose(d, 1)
    # threshold example: y = 2 relu(a) + 2 relu(b - 1.5 c), c held at 1
    W = np.array([[1.0, 0, 0], [0, 1.0, -1.5]])
    C, d = dl(W, np.zeros(2), np.array([2.0, 2.0]), 0.0, np.array([1.0, 2, 1]), np.array([0.0, 0, 1]))
    assert np.allclose(C, [2, 1, 0]) and np.isclose(d, 3)
    # exercise 1 and 2 from the DeepLIFT exercise sheet
    W = np.array([[2, 1, 0, 0], [0, -1, 1, 0], [0, 0, -2, 2]], float)
    C, d = dl(W, np.array([-1.0, 0, 2]), np.array([-1.0, 2, 1]), 0.0, np.array([1.0, 2, 0, 1]), np.zeros(4))
    assert np.allclose(C, [-1.5, -1.5, 0, 2]) and np.isclose(d, -1)
    W = np.array([[1, -1, 0, 0], [0, 2, 1, 0], [0, 0, 1, -3]], float)
    C, d = dl(W, np.array([2.0, -2, 0]), np.array([1.0, -2, 2]), 0.0, np.array([0.0, 1, 2, 1]), np.zeros(4))
    assert np.allclose(C, [0, -3, -2, 0]) and np.isclose(d, -5)

    # Shapley values of the saturation and threshold examples (reference 0)
    f_sat = lambda S: min(len(S), 1)
    assert np.allclose(shapley(f_sat, 2), [0.5, 0.5])
    xa, xb = 1.0, 2.0
    f_thr = lambda S: 2 * relu(xa * (0 in S)) + 2 * relu(xb * (1 in S) - 1.5)
    assert np.allclose(shapley(f_thr, 2), [2, 1])
    # exercise 5: y = relu(x1 + x2 - 1), x = (2, 0.5), reference 0
    C, d = dl(np.array([[1.0, 1.0]]), np.array([-1.0]), np.array([1.0]), 0.0, np.array([2.0, 0.5]), np.zeros(2))
    assert np.allclose(C, [1.2, 0.3]) and np.isclose(d, 1.5)
    xe = np.array([2.0, 0.5])
    f_ex = lambda S: relu(sum(xe[j] for j in S) - 1)
    assert np.allclose(shapley(f_ex, 2), [1.25, 0.25])
    # gradient x input for the threshold example
    assert np.allclose([2 * 1 * xa, 2 * 1 * xb], [2, 4])
    print("worked examples check out")


# ------------------------------------------------------------ simple functions

def fig_ig_saturation():
    """Integrated gradients for y = min(x1 + x2, 1) along the path from (0,0) to (1,1)."""
    a = np.linspace(0, 1, 401)
    y = np.minimum(2 * a, 1)
    g = (2 * a < 1).astype(float)
    df = pd.concat([pd.DataFrame({"alpha": a, "value": y, "curve": "output $y(\\alpha\\mathbf{x})$"}),
                    pd.DataFrame({"alpha": a, "value": g,
                                  "curve": "gradient $\\partial y/\\partial x_1$"})])
    df["curve"] = pd.Categorical(df["curve"], df["curve"].unique())
    pts = pd.DataFrame({"alpha": np.arange(6) * 0.2})
    pts["value"] = (2 * pts.alpha < 1).astype(float)
    shade = pd.DataFrame({"alpha": a, "value": g})
    p = (ggplot(df, aes("alpha", "value", color="curve"))
         + geom_area(shade, aes("alpha", "value"), inherit_aes=False, fill=PAL[1], alpha=0.15)
         + geom_line(size=0.9)
         + geom_point(pts, aes("alpha", "value"), inherit_aes=False, color=PAL[1], size=2)
         + annotate("text", x=0.25, y=0.45, label="area = 0.5", size=9, color=PAL[1])
         + scale_color_manual(values=[PAL[0], PAL[1]])
         + labs(x="path position $\\alpha$", y="")
         + THEME + theme(legend_position="top"))
    save(p, "ig_saturation.pdf", 4.2, 2.9)


def fig_secant():
    """Gradient (tangent) versus DeepLIFT multiplier (secant) for a sigmoid unit, reference 0."""
    z = np.linspace(-6, 6, 400)
    z0, z1 = 0.0, 4.0
    s0, s1 = expit(z0), expit(z1)
    grad = s1 * (1 - s1)
    mult = (s1 - s0) / (z1 - z0)
    curve = pd.DataFrame({"z": z, "y": expit(z)})
    lines = pd.concat([
        pd.DataFrame({"z": z, "y": s1 + grad * (z - z1),
                      "line": f"tangent at $z$, slope {grad:.3f} (gradient)"}),
        pd.DataFrame({"z": z, "y": s0 + mult * (z - z0),
                      "line": f"secant from $z^0$, slope {mult:.3f} (multiplier)"})])
    lines = lines[(lines.y > -0.15) & (lines.y < 1.15)]
    lines["line"] = pd.Categorical(lines["line"], lines["line"].unique())
    pts = pd.DataFrame({"z": [z0, z1], "y": [s0, s1], "lab": ["$z^0$", "$z$"]})
    p = (ggplot(curve, aes("z", "y")) + geom_line(size=1, color=PAL[4])
         + geom_line(lines, aes("z", "y", color="line", linetype="line"), size=0.7)
         + geom_point(pts, aes("z", "y"), size=2.2, color=PAL[4])
         + geom_text(pts, aes("z", "y", label="lab"), nudge_y=0.09, nudge_x=-0.35, size=9)
         + geom_segment(aes(x=z1 + 0.6, xend=z1 + 0.6, y=s0, yend=s1), color=PAL[2],
                        size=0.6, inherit_aes=False)
         + annotate("text", x=z1 + 0.75, y=(s0 + s1) / 2, label="$\\Delta y$", ha="left", size=9,
                    color=PAL[2])
         + scale_color_manual(values=[PAL[0], PAL[1]])
         + scale_linetype_manual(values=["dashed", "solid"])
         + labs(x="pre-activation $z$", y="$\\sigma(z)$")
         + THEME + theme(legend_position="top", legend_direction="vertical"))
    save(p, "secant.pdf", 4.2, 3.4)


# ------------------------------------------------------------ simulated sequences

BASES = "ACGT"
MOTIFS = {"GATA": "AGATAAG", "TAL1": "AACAGATGGT"}


def sample_motif(rng, cons, p=0.95):
    out = []
    for c in cons:
        if rng.random() < p:
            out.append(c)
        else:
            out.append(rng.choice([b for b in BASES if b != c]))
    return "".join(out)


def simulate(rng, n, L=200):
    """Positives contain both motifs; negatives contain one, the other or neither (equal thirds)."""
    seqs, labels, sites = [], [], []
    for i in range(n):
        s = list(rng.choice(list(BASES), L))
        pos = i % 2 == 0
        which = ["GATA", "TAL1"] if pos else [[], ["GATA"], ["TAL1"]][rng.integers(3)]
        st = {}
        for m in which:
            w = len(MOTIFS[m])
            while True:
                a = int(rng.integers(5, L - w - 5))
                if all(a + w + 2 <= b or b + len(MOTIFS[k]) + 2 <= a for k, b in st.items()):
                    break
            s[a:a + w] = sample_motif(rng, MOTIFS[m])
            st[m] = a
        seqs.append("".join(s)); labels.append(float(pos)); sites.append(st)
    return seqs, np.array(labels), sites


def onehot(seqs):
    idx = {b: i for i, b in enumerate(BASES)}
    X = np.zeros((len(seqs), 4, len(seqs[0])), np.float32)
    for i, s in enumerate(seqs):
        X[i, [idx[c] for c in s], np.arange(len(s))] = 1
    return torch.tensor(X)


def train_cnn():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    rng = np.random.default_rng(4)
    tr_s, ytr, _ = simulate(rng, 10000)
    te_s, yte, te_sites = simulate(rng, 1000)
    Xtr, Xte = onehot(tr_s), onehot(te_s)
    model = SeqCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    for epoch in range(40):
        perm = torch.randperm(len(Xtr))
        for k in range(0, len(Xtr), 64):
            b = perm[k:k + 64]
            opt.zero_grad()
            lossf(model(Xtr[b]), ytr_t[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        auc = roc_auc_score(yte, model(Xte).numpy())
    return model, Xte, te_s, yte, te_sites, auc


def fig_sequence_attributions():
    model, Xte, te_s, yte, te_sites, auc = train_cnn()
    print(f"test AUROC {auc:.3f}")
    with torch.no_grad():
        pred = torch.sigmoid(model(Xte)).numpy()
    # a confidently predicted positive
    i = int(np.where((yte == 1) & (pred > 0.9))[0][0])
    x = Xte[i:i + 1]
    ref = torch.full_like(x, 0.25)
    with torch.no_grad():
        f_x, f_ref = model(x).item(), model(ref).item()
    scores = {}
    d = ism(model, x)[0]
    scores["in silico mutagenesis"] = -(d.sum(0) / 3)  # f(x) minus mean over the 3 alternatives
    g = backprop_scores(model, x, "grad")
    scores["gradient $\\times$ input"] = (g * x)[0].sum(0)
    scores["guided backprop $\\times$ input"] = (backprop_scores(model, x, "guided") * x)[0].sum(0)
    scores["deconvnet $\\times$ input"] = (backprop_scores(model, x, "deconv") * x)[0].sum(0)
    print("max |guided - grad| =", (backprop_scores(model, x, "guided") - g).abs().max().item())
    C = deeplift(model, x, ref)
    scores["DeepLIFT (Rescale)"] = C[0].sum(0)
    ig = integrated_gradients(model, x, ref, steps=500)
    scores["integrated gradients"] = ig[0].sum(0)
    print(f"logit f(x)={f_x:.3f}, f(ref)={f_ref:.3f}, diff={f_x - f_ref:.3f}; "
          f"sum DeepLIFT={C.sum().item():.3f}; sum IG={ig.sum().item():.3f}; "
          f"sum grad*input={(g * x).sum().item():.3f}")
    L = x.shape[2]
    seq = te_s[i]
    rows = []
    for name, s in scores.items():
        s = s.numpy()
        rows.append(pd.DataFrame({"pos": np.arange(L), "score": s / np.abs(s).max(),
                                  "base": list(seq), "method": name}))
    df = pd.concat(rows)
    df["method"] = pd.Categorical(df["method"], list(scores))
    rect = pd.DataFrame([{"xmin": a - 0.5, "xmax": a + len(MOTIFS[m]) - 0.5, "motif": m}
                         for m, a in te_sites[i].items()])
    p = (ggplot(df, aes("pos", "score", fill="base"))
         + geom_rect(rect, aes(xmin="xmin", xmax="xmax"), ymin=-np.inf, ymax=np.inf,
                     inherit_aes=False, fill="#dddddd")
         + geom_col(width=1)
         + geom_hline(yintercept=0, size=0.3)
         + facet_wrap("~method", ncol=1)
         + scale_fill_manual(values=BASE_COL)
         + labs(x="position in sequence", y="score at observed base (scaled)")
         + THEME + theme(legend_position="top", panel_grid_major=element_blank()))
    save(p, "sequence_attributions.pdf", 6.0, 7.2)
    return dict(auc=auc, i=i, sites=te_sites[i], f_x=f_x, f_ref=f_ref)


if __name__ == "__main__":
    check_toy_examples()
    for fn in [fig_ig_saturation, fig_secant, fig_sequence_attributions]:
        res = fn()
        print("wrote", fn.__name__, res if res else "")
