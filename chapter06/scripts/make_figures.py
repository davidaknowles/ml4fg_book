"""Generate the computed figures for the hidden Markov model chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
The HMM routines (forward-backward, Viterbi, Baum-Welch) are in hmm.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, element_blank, element_text, facet_grid, facet_wrap, geom_hline,
    geom_line, geom_text, geom_tile, ggplot, labs, scale_alpha_continuous,
    scale_fill_gradient, scale_fill_manual, scale_x_continuous,
    scale_y_discrete, theme, theme_bw, guides, guide_legend, coord_cartesian,
    scale_color_identity, scale_color_manual, scale_y_log10,
)
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import (baum_welch, forward_backward, mixture_posterior,  # noqa: E402
                 simulate, stationary, viterbi)

import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

THEME = theme_bw(base_size=10) + theme(
    panel_grid_minor=element_blank(),
    strip_background=element_blank(),
    strip_text=element_text(size=9),
    legend_title=element_blank(),
)


def save(p, name, w=5.0, h=3.2):
    p.save(OUT / name, width=w, height=h, verbose=False, metadata={"CreationDate": None})


# ------------------------------------------------------------ toy chromatin HMM
# K = 5 states, M = 5 binary marks, 200 bp bins.
STATES = ["promoter", "enhancer", "transcribed", "Polycomb", "quiescent"]
MARKS = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K27me3"]
STATE_COL = ["#d1495b", "#edae49", "#66a182", "#7f7f7f", "#9ecae1"]
E_TRUE = np.array([
    # K4me3 K27ac K4me1 K36me3 K27me3
    [0.75, 0.60, 0.40, 0.05, 0.05],   # promoter
    [0.10, 0.45, 0.70, 0.05, 0.03],   # enhancer
    [0.03, 0.05, 0.10, 0.55, 0.03],   # transcribed
    [0.10, 0.03, 0.10, 0.03, 0.55],   # Polycomb
    [0.03, 0.03, 0.05, 0.05, 0.05],   # quiescent
])
T_TRUE = np.array([
    [0.80, 0.02, 0.15, 0.01, 0.02],
    [0.08, 0.80, 0.02, 0.01, 0.09],
    [0.01, 0.01, 0.96, 0.00, 0.02],
    [0.01, 0.01, 0.00, 0.96, 0.02],
    [0.004, 0.008, 0.002, 0.006, 0.98],
])
PI_TRUE = stationary(T_TRUE)


def fig_hmm_posterior():
    """Simulated binary marks, true states, HMM and mixture posteriors, Viterbi path."""
    rng = np.random.default_rng(6)
    S = 20000
    x, Y = simulate(PI_TRUE, T_TRUE, E_TRUE, S, rng)
    gamma, _, _ = forward_backward(Y, PI_TRUE, T_TRUE, E_TRUE)
    mix = mixture_posterior(Y, PI_TRUE, E_TRUE)
    vit = viterbi(Y, PI_TRUE, T_TRUE, E_TRUE)
    acc = {"HMM posterior": (gamma.argmax(1) == x).mean(),
           "mixture posterior": (mix.argmax(1) == x).mean(),
           "Viterbi": (vit == x).mean()}
    print("state accuracy over", S, "bins:", {k: round(v, 3) for k, v in acc.items()})

    # window to display: choose one containing every state
    W, start = 300, 0
    for s0 in range(0, S - W, 50):
        if len(set(x[s0:s0 + W])) == len(STATES):
            start = s0
            break
    sl = slice(start, start + W)
    pos = np.arange(W) + 1
    panels = ["marks", "true state", "HMM\nposterior", "Viterbi", "mixture\nposterior"]
    rows = []
    for m, mk in enumerate(MARKS):
        on = Y[sl, m] == 1
        rows.append(pd.DataFrame({"t": pos[on], "y": mk, "panel": panels[0],
                                  "state": None, "a": 1.0}))
    rows.append(pd.DataFrame({"t": pos, "y": " ", "panel": panels[1],
                              "state": [STATES[k] for k in x[sl]], "a": 1.0}))
    rows.append(pd.DataFrame({"t": pos, "y": "  ", "panel": panels[3],
                              "state": [STATES[k] for k in vit[sl]], "a": 1.0}))
    for P, name in [(gamma, panels[2]), (mix, panels[4])]:
        for k, st in enumerate(STATES):
            rows.append(pd.DataFrame({"t": pos, "y": st + " ", "panel": name,
                                      "state": st, "a": P[sl, k]}))
    df = pd.concat(rows, ignore_index=True)
    df["panel"] = pd.Categorical(df["panel"], panels)
    df["state"] = pd.Categorical(df["state"], STATES)
    ylev = MARKS[::-1] + [" ", "  "] + [s + " " for s in STATES[::-1]]
    df["y"] = pd.Categorical(df["y"], ylev)
    marks = df[df.panel == panels[0]]
    rest = df[df.panel != panels[0]]
    p = (ggplot()
         + geom_tile(marks, aes("t", "y"), fill="black", width=1, height=0.8)
         + geom_tile(rest, aes("t", "y", fill="state", alpha="a"), width=1, height=0.95)
         + facet_grid("panel ~ .", scales="free_y", space={"x": [1], "y": [5, 1, 5, 1, 5]})
         + scale_fill_manual(values=STATE_COL, drop=False)
         + scale_alpha_continuous(range=(0, 1), limits=(0, 1), guide=None)
         + scale_x_continuous(expand=(0, 0))
         + guides(fill=guide_legend(nrow=1))
         + labs(x="genomic bin (200 bp)", y="")
         + THEME + theme(strip_text_y=element_text(angle=0, size=8),
                         legend_position="bottom", axis_text_y=element_text(size=7),
                         panel_grid_major=element_blank()))
    save(p, "hmm_posterior.pdf", 6.2, 4.6)
    return acc


def match_states(E_hat, E_ref):
    """Permutation of estimated states that best matches the reference emissions."""
    cost = ((E_hat[:, None, :] - E_ref[None, :, :]) ** 2).sum(-1)
    row, col = linear_sum_assignment(cost)
    perm = np.empty(len(col), dtype=int)
    perm[col] = row
    return perm


def fig_baum_welch():
    """Log-likelihood traces from random initializations and the recovered parameters."""
    rng = np.random.default_rng(11)
    S = 20000
    _, Y = simulate(PI_TRUE, T_TRUE, E_TRUE, S, rng)
    _, _, ll_true = forward_backward(Y, PI_TRUE, T_TRUE, E_TRUE)
    K = len(STATES)
    fits, traces = [], []
    for r in range(8):
        pi, T, E, tr = baum_welch(Y, K, np.random.default_rng(100 + r), n_iter=150)
        fits.append((pi, T, E, tr[-1]))
        traces.append(pd.DataFrame({"iteration": np.arange(1, len(tr) + 1),
                                    "ll": tr / S, "run": str(r)}))
    final = np.array([f[3] for f in fits]) / S
    print("true-parameter log lik per bin:", round(ll_true / S, 4))
    print("final log lik per bin by run:", np.round(np.sort(final), 4))
    tdf = pd.concat(traces, ignore_index=True)
    p = (ggplot(tdf, aes("iteration", "ll", group="run"))
         + geom_line(color="#1b6ca8", alpha=0.8, size=0.5)
         + geom_hline(yintercept=ll_true / S, linetype="dashed", color="#d1495b")
         + labs(x="EM iteration", y="log likelihood per bin")
         + THEME)
    lo = np.quantile(tdf.ll, 0.03)
    p = p + coord_cartesian(ylim=(lo, max(tdf.ll.max(), ll_true / S) + 0.005))
    save(p, "baum_welch_trace.pdf", 4.6, 3.0)

    # parameters of the best run, states matched to the truth
    best = int(np.argmax(final))
    _, T_hat, E_hat, _ = fits[best]
    perm = match_states(E_hat, E_TRUE)
    E_hat, T_hat = E_hat[perm], T_hat[np.ix_(perm, perm)]
    print("max abs error E:", round(np.abs(E_hat - E_TRUE).max(), 3),
          " T:", round(np.abs(T_hat - T_TRUE).max(), 3))
    rows = []
    for name, A, cols in [("emission, true", E_TRUE, MARKS), ("emission, estimated", E_hat, MARKS),
                          ("transition, true", T_TRUE, STATES),
                          ("transition, estimated", T_hat, STATES)]:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                rows.append({"panel": name, "from": STATES[i], "to": cols[j], "v": A[i, j],
                             "lab": f"{A[i, j]:.2f}".lstrip("0")})
    df = pd.DataFrame(rows)
    df["panel"] = pd.Categorical(df["panel"], ["emission, true", "emission, estimated",
                                               "transition, true", "transition, estimated"])
    df["from"] = pd.Categorical(df["from"], STATES[::-1])
    df["to"] = pd.Categorical(df["to"], MARKS + STATES)
    df["txtcol"] = np.where(df.v > 0.85, "white", "black")
    p = (ggplot(df, aes("to", "from", fill="v"))
         + geom_tile(color="white")
         + geom_text(aes(label="lab", color="txtcol"), size=7.5)
         + facet_wrap("~panel", ncol=2, scales="free_x")
         + scale_fill_gradient(low="#f7fbff", high="#08306b", limits=(0, 1), name="")
         + scale_color_identity()
         + labs(x="", y="state")
         + THEME + theme(axis_text_x=element_text(angle=45, ha="right", size=8),
                         axis_text_y=element_text(size=8), panel_grid_major=element_blank(),
                         legend_position="none"))
    save(p, "baum_welch_params.pdf", 6.0, 5.4)


def fig_dwell():
    """Geometric distribution of segment lengths for several self-transition probabilities."""
    rows = []
    for a in [0.8, 0.95, 0.99]:
        L = np.arange(1, 151)
        rows.append(pd.DataFrame({"L": L, "p": a ** (L - 1) * (1 - a),
                                  "a": f"$T_{{ii}}={a}$, mean {1 / (1 - a):.0f}"}))
    df = pd.concat(rows)
    df = df[df.p > 1e-5]
    p = (ggplot(df, aes("L", "p", color="a")) + geom_line(size=0.8)
         + scale_y_log10()
         + scale_color_manual(values=["#1b6ca8", "#d1495b", "#66a182"])
         + labs(x="segment length $L$ (bins)", y="$P(L)$")
         + THEME + theme(legend_position="right"))
    save(p, "dwell_time.pdf", 5.0, 2.5)


if __name__ == "__main__":
    which = sys.argv[1:] or ["posterior", "baum_welch", "dwell"]
    if "posterior" in which:
        fig_hmm_posterior()
    if "baum_welch" in which:
        fig_baum_welch()
    if "dwell" in which:
        fig_dwell()
