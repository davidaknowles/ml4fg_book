"""Generate the computed figures for the recurrent neural networks chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, element_blank, element_text, facet_wrap, geom_hline, geom_line,
    geom_vline, ggplot, labs, scale_color_manual, scale_linetype_manual,
    theme, theme_bw,
)
from scipy.special import expit
from scipy.stats import beta as beta_dist

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


# ---------------------------------------------------------------- activations

def fig_tanh():
    """tanh and the logistic sigmoid, and their derivatives."""
    x = np.linspace(-5, 5, 400)
    t, s = np.tanh(x), expit(x)
    df = pd.concat([
        pd.DataFrame({"x": x, "y": t, "fn": "tanh", "panel": "function"}),
        pd.DataFrame({"x": x, "y": s, "fn": "sigmoid", "panel": "function"}),
        pd.DataFrame({"x": x, "y": 1 - t ** 2, "fn": "tanh", "panel": "derivative"}),
        pd.DataFrame({"x": x, "y": s * (1 - s), "fn": "sigmoid", "panel": "derivative"}),
    ])
    df["panel"] = pd.Categorical(df["panel"], ["function", "derivative"])
    df["fn"] = pd.Categorical(df["fn"], ["tanh", "sigmoid"])
    p = (ggplot(df, aes("x", "y", color="fn", linetype="fn")) + geom_line(size=0.8)
         + geom_hline(yintercept=0, color="grey") + geom_vline(xintercept=0, color="grey")
         + facet_wrap("~panel", scales="free_y")
         + scale_color_manual(values=PAL[:2]) + scale_linetype_manual(values=["solid", "dashed"])
         + labs(x="$a$", y="") + THEME)
    save(p, "tanh.pdf", 5.5, 2.5)


# ---------------------------------------------------------------- gradient flow

def orthogonal(rng, k):
    q, r = np.linalg.qr(rng.normal(size=(k, k)))
    return q * np.sign(np.diag(r))


def vanilla_jacobian_norms(rng, scale, K=50, P=10, S=101):
    """Spectral norm of d h_S / d h_{S-k} for a tanh RNN whose W_h has all singular values = scale."""
    Wh = scale * orthogonal(rng, K)
    Wx = rng.normal(0, 1 / np.sqrt(P), (K, P))
    b = np.zeros(K)
    x = rng.normal(size=(S, P))
    h = np.zeros(K)
    jacs = []
    for t in range(S):
        h = np.tanh(Wh @ h + Wx @ x[t] + b)
        jacs.append((1 - h ** 2)[:, None] * Wh)       # d h_t / d h_{t-1}
    J = np.eye(K)
    norms = [1.0]
    for k in range(1, S):
        J = J @ jacs[S - k]                             # d h_S / d h_{S-k}
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)


def gru_jacobian_norms(rng, bz, K=50, P=10, S=101):
    """Spectral norm of d h_S / d h_{S-k} for a GRU (slide parameterization) with update-gate bias bz.

    z = sigmoid(Wz [x; h] + bz), r = sigmoid(Wr [x; h]), g = tanh(Wx x + r * (Wh h)),
    h_new = z * g + (1 - z) * h.
    """
    sd = 1 / np.sqrt(K + P)
    Wzx, Wzh = rng.normal(0, sd, (K, P)), rng.normal(0, sd, (K, K))
    Wrx, Wrh = rng.normal(0, sd, (K, P)), rng.normal(0, sd, (K, K))
    Wx = rng.normal(0, 1 / np.sqrt(P), (K, P))
    Wh = orthogonal(rng, K)
    x = rng.normal(size=(S, P))
    h = np.zeros(K)
    jacs = []
    for t in range(S):
        z = expit(Wzx @ x[t] + Wzh @ h + bz)
        r = expit(Wrx @ x[t] + Wrh @ h)
        u = Wh @ h
        g = np.tanh(Wx @ x[t] + r * u)
        dz = (z * (1 - z))[:, None] * Wzh
        dr = (r * (1 - r))[:, None] * Wrh
        dg = (1 - g ** 2)[:, None] * (r[:, None] * Wh + u[:, None] * dr)
        jac = np.diag(1 - z) + (g - h)[:, None] * dz + z[:, None] * dg
        jacs.append(jac)
        h = z * g + (1 - z) * h
    J = np.eye(K)
    norms = [1.0]
    for k in range(1, S):
        J = J @ jacs[S - k]
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)


def fig_gradient_flow():
    rows = []
    for s in [0.9, 1.5, 3.0]:
        n = vanilla_jacobian_norms(np.random.default_rng(1), s)
        rows.append(pd.DataFrame({"lag": np.arange(n.size), "norm": n,
                                  "setting": f"$\\sigma(\\mathbf{{W}}_h)={s:g}$", "panel": "vanilla RNN"}))
    for bz in [0.0, -3.0, -6.0]:
        n = gru_jacobian_norms(np.random.default_rng(2), bz)
        rows.append(pd.DataFrame({"lag": np.arange(n.size), "norm": n,
                                  "setting": f"$b_z={bz:g}$", "panel": "GRU"}))
    df = pd.concat(rows)
    df["panel"] = pd.Categorical(df["panel"], ["vanilla RNN", "GRU"])
    df["setting"] = pd.Categorical(df["setting"], df["setting"].unique())
    df["log10norm"] = np.log10(df["norm"])
    p = (ggplot(df, aes("lag", "log10norm", color="setting")) + geom_line(size=0.8)
         + geom_hline(yintercept=0, color="grey", linetype="dotted")
         + facet_wrap("~panel")
         + scale_color_manual(values=[PAL[0], PAL[4], PAL[1], PAL[2], PAL[3], "#8e5ea2"])
         + labs(x="lag $k$", y="$\\log_{10}\\|\\partial\\mathbf{h}_S/\\partial\\mathbf{h}_{S-k}\\|_2$")
         + THEME)
    save(p, "gradient_flow.pdf", 6.0, 2.8)
    return df


# ---------------------------------------------------------------- methylation

def fig_methylation_posterior():
    """Posterior of the methylation level of one CpG given m methylated reads out of n, uniform prior."""
    q = np.linspace(0, 1, 500)
    rows = []
    for n, m in [(1, 1), (5, 4), (30, 21), (200, 141)]:
        rows.append(pd.DataFrame({"q": q, "density": beta_dist.pdf(q, 1 + m, 1 + n - m),
                                  "reads": f"{m}/{n} reads methylated"}))
    df = pd.concat(rows)
    df["reads"] = pd.Categorical(df["reads"], df["reads"].unique())
    p = (ggplot(df, aes("q", "density", color="reads")) + geom_line(size=0.8)
         + geom_vline(xintercept=0.7, color="grey", linetype="dashed")
         + scale_color_manual(values=PAL[:4])
         + labs(x="methylation level $q$", y="posterior density") + THEME)
    save(p, "methylation_posterior.pdf", 4.6, 2.6)


if __name__ == "__main__":
    fig_tanh()
    df = fig_gradient_flow()
    print(df.groupby(["panel", "setting"], observed=True)["log10norm"].agg(["first", lambda v: v.iloc[10], lambda v: v.iloc[50], "last"]))
    fig_methylation_posterior()
