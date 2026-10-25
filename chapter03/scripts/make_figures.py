"""Generate the computed figures for the chromatin and transformers chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, annotate, element_blank, element_text, facet_wrap, geom_area,
    geom_col, geom_hline, geom_line, geom_point, geom_tile, geom_vline,
    ggplot, labs, scale_color_manual, scale_fill_gradient,
    scale_fill_manual, scale_linetype_manual, scale_x_continuous,
    scale_x_log10, scale_y_continuous, scale_y_log10, scale_y_reverse,
    coord_equal, theme, theme_bw,
)
from scipy.special import expit, softmax

import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
PAL = ["#1b6ca8", "#d1495b", "#66a182", "#edae49", "#3d3d3d"]

THEME = theme_bw(base_size=10) + theme(
    panel_grid_minor=element_blank(),
    strip_background=element_blank(),
    strip_text=element_text(size=10),
    legend_title=element_blank(),
)


def save(p, name, w=5.0, h=3.2):
    p.save(OUT / name, width=w, height=h, verbose=False, metadata={"CreationDate": None})


# ------------------------------------------------------------ chromatin assays

def fig_assay_profiles():
    """Simulated read coverage around a bound TF in a nucleosome-depleted region.

    TF ChIP-seq reads come from a narrow window around the bound site. DNase/ATAC
    cuts fall in the accessible linker DNA, except where the bound TF protects the
    DNA (a footprint). H3K27ac ChIP-seq reads come from the acetylated nucleosomes
    flanking the nucleosome-depleted region.
    """
    rng = np.random.default_rng(3)
    x = np.arange(-1000, 1001)
    # TF ChIP: fragments centred on the binding site
    tf = np.exp(-0.5 * (x / 70) ** 2)
    # accessibility: open between the two flanking nucleosomes, footprint at the TF
    open_ = expit((x + 170) / 15) * expit((170 - x) / 15)
    footprint = 1 - 0.85 * np.exp(-0.5 * (x / 12) ** 2)
    linker = 0.05 * (1 + np.cos(2 * np.pi * (np.abs(x) - 350) / 180)) * (np.abs(x) > 300)  # linker DNA between nucleosomes
    access = open_ * footprint + linker
    # H3K27ac: acetylated nucleosomes at +-260, +-440, +-620 with decaying signal
    hist = np.zeros_like(x, dtype=float)
    for c, a in [(260, 1.0), (440, 0.7), (620, 0.45), (800, 0.25)]:
        hist += a * (np.exp(-0.5 * ((x - c) / 55) ** 2) + np.exp(-0.5 * ((x + c) / 55) ** 2))
    rows = []
    names = ["TF ChIP-seq", "DNase-seq / ATAC-seq", "H3K27ac ChIP-seq"]
    for name, dens, n in zip(names, [tf, access, hist], [3000, 6000, 6000]):
        pos = rng.choice(x, size=n, p=dens / dens.sum())
        counts, edges = np.histogram(pos, bins=np.arange(-1000, 1011, 10))
        rows.append(pd.DataFrame({"x": edges[:-1] + 5, "y": counts / counts.max(), "assay": name}))
    df = pd.concat(rows)
    df["assay"] = pd.Categorical(df["assay"], categories=names)
    p = (ggplot(df, aes("x", "y", fill="assay")) + geom_area(alpha=0.85)
         + facet_wrap("~assay", ncol=1)
         + scale_fill_manual(values=[PAL[1], "#5b3a8e", PAL[2]], guide=None)
         + geom_vline(xintercept=0, linetype="dotted", color="grey")
         + labs(x="position relative to TF binding site (bp)", y="normalized read count")
         + THEME)
    save(p, "assay_profiles.pdf", 5.0, 4.2)


# ------------------------------------------------------------ eve stripe 2

def fig_eve_stripe2():
    """Toy model of the even-skipped stripe 2 enhancer.

    Activators (Bicoid, Hunchback) are high anteriorly; repressors (Giant, anterior
    domain, Kruppel central domain) carve out a stripe. Enhancer activity is a
    logistic function of weighted activator minus repressor input.
    """
    x = np.linspace(0, 100, 501)  # % egg length from anterior
    bcd = np.exp(-x / 22)
    hb = expit((50 - x) / 3)
    gt = expit((30 - x) / 2.5)  # anterior domain, simplified to extend to the tip
    kr = np.exp(-0.5 * ((x - 53) / 6) ** 2)
    act = 0.6 * bcd + 0.6 * hb
    rep = 1.2 * gt + 1.2 * kr
    out = expit(25 * (act - rep - 0.45))
    tfs = pd.DataFrame({"x": np.tile(x, 4),
                        "y": np.concatenate([bcd, hb, gt, kr]),
                        "factor": np.repeat(["Bicoid (activator)", "Hunchback (activator)",
                                             "Giant (repressor)", "Kruppel (repressor)"], x.size),
                        "panel": "transcription factor concentration"})
    order = ["Bicoid (activator)", "Hunchback (activator)", "Giant (repressor)", "Kruppel (repressor)"]
    tfs["factor"] = pd.Categorical(tfs["factor"], categories=order)
    outdf = pd.DataFrame({"x": x, "y": out, "panel": "stripe 2 enhancer activity"})
    panels = ["transcription factor concentration", "stripe 2 enhancer activity"]
    tfs["panel"] = pd.Categorical(tfs["panel"], categories=panels)
    outdf["panel"] = pd.Categorical(outdf["panel"], categories=panels)
    p = (ggplot() + geom_line(tfs, aes("x", "y", color="factor", linetype="factor"), size=0.8)
         + geom_area(outdf, aes("x", "y"), fill=PAL[3], alpha=0.8)
         + facet_wrap("~panel", ncol=1, scales="free_y")
         + scale_color_manual(values=[PAL[0], PAL[0], PAL[1], PAL[1]])
         + scale_linetype_manual(values=["solid", "dashed", "solid", "dashed"])
         + labs(x="position along the embryo (% egg length, anterior = 0)", y="")
         + THEME + theme(legend_position="right"))
    save(p, "eve_stripe2.pdf", 6.0, 3.8)
    lo, hi = x[out > 0.5].min(), x[out > 0.5].max()
    print(f"eve stripe 2 toy model: active between {lo:.1f}% and {hi:.1f}% egg length")


# ------------------------------------------------------------ receptive field

def receptive_field(depth, k, pool=1, dilate=False):
    """Receptive field (in input positions) after each of `depth` conv layers."""
    rf, jump, out = 1, 1, []
    for d in range(depth):
        dil = 2 ** d if dilate else 1
        rf += (k - 1) * dil * jump
        if pool > 1:
            rf += (pool - 1) * jump
            jump *= pool
        out.append(rf)
    return np.array(out)


def fig_receptive_field():
    D = np.arange(1, 16)
    k = 5
    df = pd.concat([
        pd.DataFrame({"depth": D, "rf": receptive_field(15, k), "arch": "convolution"}),
        pd.DataFrame({"depth": D, "rf": receptive_field(15, k, pool=2),
                      "arch": "convolution + max pool (2)"}),
        pd.DataFrame({"depth": D, "rf": receptive_field(15, k, dilate=True),
                      "arch": "dilated convolution"}),
    ])
    order = ["convolution", "convolution + max pool (2)", "dilated convolution"]
    df["arch"] = pd.Categorical(df["arch"], categories=order)
    p = (ggplot(df, aes("depth", "rf", color="arch")) + geom_line() + geom_point(size=1.2)
         + geom_hline(yintercept=2e5, linetype="dashed", color="grey")
         + annotate("text", x=1, y=3.2e5, label="200 kb", ha="left", size=8, color="grey")
         + geom_hline(yintercept=1e3, linetype="dashed", color="grey")
         + annotate("text", x=1, y=1.6e3, label="1 kb", ha="left", size=8, color="grey")
         + scale_y_log10() + scale_color_manual(values=PAL[:3])
         + labs(x="number of convolutional layers $D$", y="receptive field (bp)")
         + THEME + theme(legend_position="bottom"))
    save(p, "receptive_field.pdf", 5.0, 3.4)


# ------------------------------------------------------------ attention

def fig_attention_matrices():
    """A width-3 convolution as a fixed banded weight pattern vs an attention matrix."""
    rng = np.random.default_rng(7)
    L, D, P = 12, 8, 8
    conv = np.zeros((L, L))
    for i in range(L):
        for j in range(max(0, i - 1), min(L, i + 2)):
            conv[i, j] = 1.0
    conv /= conv.sum(1, keepdims=True)
    H = rng.normal(size=(L, D))
    Wq, Wk = rng.normal(size=(P, D)) / np.sqrt(D), rng.normal(size=(P, D)) / np.sqrt(D)
    Q, K = H @ Wq.T, H @ Wk.T
    A = softmax(2.0 * Q @ K.T / np.sqrt(P), axis=1)
    rows = []
    for name, M in [("convolution (fixed, local)", conv), ("self-attention (input-dependent, global)", A)]:
        ii, jj = np.meshgrid(np.arange(1, L + 1), np.arange(1, L + 1), indexing="ij")
        rows.append(pd.DataFrame({"i": ii.ravel(), "j": jj.ravel(), "w": M.ravel(), "type": name}))
    df = pd.concat(rows)
    df["type"] = pd.Categorical(df["type"], categories=["convolution (fixed, local)",
                                                        "self-attention (input-dependent, global)"])
    p = (ggplot(df, aes("j", "i", fill="w")) + geom_tile(color="white", size=0.2)
         + facet_wrap("~type") + scale_y_reverse(breaks=[1, 4, 8, 12])
         + scale_x_continuous(breaks=[1, 4, 8, 12])
         + scale_fill_gradient(low="#f7f7f7", high=PAL[0], name="weight")
         + coord_equal()
         + labs(x="input position $j$ (previous layer)", y="output position $i$ (next layer)")
         + THEME + theme(legend_title=element_text(size=9)))
    save(p, "attention_matrices.pdf", 6.0, 3.2)


def fig_attention_scaling():
    """Largest attention weight with and without the 1/sqrt(P) scaling."""
    rng = np.random.default_rng(11)
    L, reps = 50, 2000
    Ps = 2 ** np.arange(1, 11)
    rows = []
    for P in Ps:
        q = rng.normal(size=(reps, 1, P))
        k = rng.normal(size=(reps, L, P))
        s = (k @ q.transpose(0, 2, 1))[..., 0]  # reps x L scores
        for name, sc in [("unscaled $q^\\top k$", s), ("scaled $q^\\top k/\\sqrt{P}$", s / np.sqrt(P))]:
            a = softmax(sc, axis=1)
            rows.append({"P": P, "max": a.max(1).mean(), "scores": name})
    df = pd.DataFrame(rows)
    p = (ggplot(df, aes("P", "max", color="scores")) + geom_line() + geom_point(size=1.2)
         + geom_hline(yintercept=1 / L, linetype="dotted", color="grey")
         + annotate("text", x=2, y=0.06, label="uniform, $1/L$", ha="left", size=8, color="grey")
         + scale_x_log10(breaks=[2, 8, 32, 128, 512])
         + scale_color_manual(values=[PAL[1], PAL[0]])
         + labs(x="query/key dimension $P$", y="mean largest attention weight")
         + THEME + theme(legend_position="bottom"))
    save(p, "attention_scaling.pdf", 4.6, 3.2)


def fig_positional_encoding():
    L, D, b = 60, 32, 10000.0
    i = np.arange(L)[:, None]
    m = np.arange(D // 2)[None, :]
    omega = 1.0 / b ** (2 * m / D)
    Pm = np.zeros((L, D))
    Pm[:, 0::2] = np.sin(i * omega)
    Pm[:, 1::2] = np.cos(i * omega)
    ii, cc = np.meshgrid(np.arange(L), np.arange(D), indexing="ij")
    df = pd.DataFrame({"pos": ii.ravel(), "ch": cc.ravel(), "v": Pm.ravel()})
    p1 = (ggplot(df, aes("ch", "pos", fill="v")) + geom_tile()
          + scale_y_reverse() + scale_fill_gradient(low="#f7fbff", high="#08306b", name="")
          + labs(x="channel", y="position $i$") + THEME)
    save(p1, "positional_encoding.pdf", 3.0, 3.4)
    delta = np.arange(-60, 61)
    w = 1.0 / b ** (2 * np.arange(D // 2) / D)
    dot = np.cos(delta[:, None] * w[None, :]).sum(1)
    # check that p_i . p_{i+delta} does not depend on i
    for i0 in [0, 17, 40]:
        j = np.arange(L)
        sel = (i0 + delta >= 0) & (i0 + delta < L)
        direct = Pm[i0] @ Pm[(i0 + delta[sel])].T
        assert np.allclose(direct, dot[sel])
    df2 = pd.DataFrame({"delta": delta, "dot": dot})
    p2 = (ggplot(df2, aes("delta", "dot")) + geom_line(color=PAL[0])
          + labs(x="offset $\\Delta$", y="$p_i^\\top p_{i+\\Delta}$") + THEME)
    save(p2, "positional_dot.pdf", 3.0, 3.4)


# ------------------------------------------------------------ Hi-C and TADs

def fig_hic_tads():
    rng = np.random.default_rng(5)
    N = 200  # 10 kb bins -> 2 Mb
    sizes = [22, 38, 15, 45, 30, 50]
    bounds = np.cumsum([0] + sizes)
    tad = np.zeros(N, dtype=int)
    for t in range(len(sizes)):
        tad[bounds[t]:bounds[t + 1]] = t
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    dist = np.abs(ii - jj)
    expected = 400.0 / (dist + 1) * (1 + 6 * (tad[ii] == tad[jj]))
    upper = rng.poisson(np.triu(expected))
    C = np.triu(upper) + np.triu(upper, 1).T
    df = pd.DataFrame({"i": ii.ravel() * 10, "j": jj.ravel() * 10, "c": np.log10(1 + C.ravel())})
    p1 = (ggplot(df, aes("j", "i", fill="c")) + geom_tile()
          + scale_y_reverse() + coord_equal()
          + scale_fill_gradient(low="#fff5f0", high="#a50f15", name="log10(1+count)")
          + labs(x="genomic position (kb)", y="genomic position (kb)")
          + THEME + theme(legend_position="bottom", legend_title=element_text(size=8),
                          legend_text=element_text(size=7)))
    save(p1, "hic_map.pdf", 3.4, 3.9)
    # directionality index (Dixon et al. 2012), window of w bins
    w = 15
    di = np.zeros(N)
    for i in range(w, N - w):  # skip the edges, where one side is truncated
        A = C[i, i - w:i].sum()
        B = C[i, i + 1:i + 1 + w].sum()
        E = (A + B) / 2
        if E > 0 and A != B:
            di[i] = np.sign(B - A) * ((A - E) ** 2 / E + (B - E) ** 2 / E)
    df2 = pd.DataFrame({"pos": np.arange(N) * 10, "di": di,
                        "sign": np.where(di >= 0, "downstream", "upstream")})
    p2 = (ggplot(df2, aes("pos", "di", fill="sign")) + geom_col(width=10)
          + geom_vline(xintercept=bounds[1:-1] * 10 - 5, linetype="dashed", color="grey")
          + scale_fill_manual(values=[PAL[1], PAL[2]], guide=None)
          + labs(x="genomic position (kb)", y="directionality index") + THEME)
    save(p2, "hic_di.pdf", 3.4, 3.9)


if __name__ == "__main__":
    fig_assay_profiles()
    fig_eve_stripe2()
    fig_receptive_field()
    fig_attention_matrices()
    fig_attention_scaling()
    fig_positional_encoding()
    fig_hic_tads()
