"""Generate the computed figures for the regression chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from contourpy import contour_generator
from plotnine import (
    aes, annotate, element_blank, element_text, facet_wrap, geom_abline,
    geom_area, geom_hline, geom_line, geom_path, geom_point, geom_polygon,
    geom_ribbon, geom_segment, geom_text, geom_vline, ggplot, labs,
    scale_color_manual, scale_linetype_manual, scale_shape_manual,
    scale_x_log10, scale_y_log10, theme, theme_bw, coord_equal, xlim, ylim,
    arrow,
)
from scipy.special import expit
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

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


def contour_paths(xx, yy, zz, levels):
    """Contour lines of zz on the grid as a long data frame (level, piece, x, y)."""
    gen = contour_generator(xx, yy, zz)
    rows = []
    for lev in levels:
        for k, seg in enumerate(gen.lines(lev)):
            rows.append(pd.DataFrame({"x": seg[:, 0], "y": seg[:, 1],
                                      "level": lev, "piece": f"{lev}_{k}"}))
    return pd.concat(rows, ignore_index=True)


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


# ---------------------------------------------------------------- regression

def fig_linreg_scatter():
    rng = np.random.default_rng(1)
    x = rng.uniform(-20, 60, 100)
    y = 5 + 0.15 * x + rng.normal(0, 1.5, x.size)
    b1, b0 = np.polyfit(x, y, 1)
    df = pd.DataFrame({"x": x, "y": y})
    p = (ggplot(df, aes("x", "y")) + geom_point(color=PAL[0], size=1.2)
         + geom_abline(intercept=b0, slope=b1, color=PAL[1], size=0.8)
         + labs(x="input / feature / covariate $x$ (e.g. gene expression)",
                y="response $y$ (e.g. drug sensitivity)") + THEME)
    save(p, "linreg_scatter.pdf", 4.5, 3.0)


def correlated_design(rng, n=50, rho=0.98):
    cov = np.array([[1, rho], [rho, 1]])
    X = rng.multivariate_normal([0, 0], cov, n)
    X = (X - X.mean(0)) / X.std(0)
    y = X @ np.array([1.0, 1.0]) + rng.normal(0, 1, n)
    return X, y - y.mean()


def fig_colinearity_contours():
    """Sum-of-squares contours for two highly correlated covariates, with and without a ridge penalty."""
    rng = np.random.default_rng(3)
    X, y = correlated_design(rng)
    lam = 20.0
    g = np.linspace(-3, 5, 300)
    b1, b2 = np.meshgrid(g, g)
    B = np.stack([b1.ravel(), b2.ravel()], 1)
    ss = ((y[:, None] - X @ B.T) ** 2).sum(0).reshape(b1.shape)
    pen = ss + lam * (b1 ** 2 + b2 ** 2)
    out, opt = [], []
    for name, z, bhat in [("least squares", ss, ridge(X, y, 0)),
                          (f"ridge, $\\lambda={lam:g}$", pen, ridge(X, y, lam))]:
        zmin = z.min()
        levels = zmin + np.array([2, 8, 25, 60, 120, 200])
        c = contour_paths(b1, b2, z, levels)
        c["panel"] = name
        out.append(c)
        opt.append(pd.DataFrame({"x": [bhat[0]], "y": [bhat[1]], "panel": [name]}))
    c, o = pd.concat(out), pd.concat(opt)
    p = (ggplot(c, aes("x", "y", group="piece")) + geom_path(color=PAL[0], size=0.5)
         + geom_point(o, aes("x", "y"), inherit_aes=False, color=PAL[1], size=2)
         + geom_hline(yintercept=0, color="grey") + geom_vline(xintercept=0, color="grey")
         + facet_wrap("~panel") + coord_equal()
         + labs(x="$\\beta_1$", y="$\\beta_2$") + THEME)
    save(p, "colinearity_contours.pdf", 5.5, 2.9)


def fig_prior_likelihood_posterior():
    """One coefficient, Gaussian prior and Gaussian likelihood: the posterior is their normalized product."""
    rng = np.random.default_rng(2)
    n, sigma, sigma_b = 5, 1.0, 0.5
    x = rng.normal(0, 1, n)
    y = 1.2 * x + rng.normal(0, sigma, n)
    b = np.linspace(-1.5, 3, 600)
    prior = np.exp(-b ** 2 / (2 * sigma_b ** 2))
    lik = np.exp(-((y[:, None] - x[:, None] * b) ** 2).sum(0) / (2 * sigma ** 2))
    post = prior * lik
    norm = lambda f: f / np.trapz(f, b)
    df = pd.concat([pd.DataFrame({"beta": b, "density": norm(f), "curve": nm})
                    for nm, f in [("prior", prior), ("likelihood (normalized)", lik),
                                  ("posterior", post)]])
    df["curve"] = pd.Categorical(df["curve"], ["prior", "likelihood (normalized)", "posterior"])
    mle = (x @ y) / (x @ x)
    mapb = (x @ y) / (x @ x + sigma ** 2 / sigma_b ** 2)
    p = (ggplot(df, aes("beta", "density", color="curve")) + geom_line(size=0.8)
         + geom_vline(xintercept=mle, linetype="dashed", color=PAL[1])
         + geom_vline(xintercept=mapb, linetype="dashed", color=PAL[2])
         + annotate("text", x=mle + 0.05, y=df.density.max(), label="MLE", ha="left", size=8)
         + annotate("text", x=mapb - 0.05, y=df.density.max(), label="MAP", ha="right", size=8)
         + scale_color_manual(values=[PAL[4], PAL[1], PAL[2]])
         + labs(x="$\\beta$", y="density") + THEME + theme(legend_position="top"))
    save(p, "prior_likelihood_posterior.pdf", 4.5, 3.0)


def sim_highdim(rng, n=60, p=100, sigma=1.0):
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:10] = rng.normal(0, 1, 10)
    beta[10:] = rng.normal(0, 0.1, p - 10)
    y = X @ beta + rng.normal(0, sigma, n)
    return X, y, beta


def fig_bias_variance():
    """Exact bias^2 and variance of ridge for a fixed design, as a function of lambda."""
    rng = np.random.default_rng(4)
    sigma = 1.0
    X, _, beta = sim_highdim(rng, n=80, p=60, sigma=sigma)
    p = X.shape[1]
    lams = np.logspace(-2, 3, 80)
    G = X.T @ X
    rows = []
    for lam in lams:
        Ainv = np.linalg.inv(G + lam * np.eye(p))
        bias2 = np.sum((((Ainv @ G) - np.eye(p)) @ beta) ** 2)
        var = sigma ** 2 * np.trace(Ainv @ G @ Ainv)
        rows += [(lam, "bias$^2$", bias2), (lam, "variance", var),
                 (lam, "generalization error", bias2 + var + sigma ** 2)]
    df = pd.DataFrame(rows, columns=["lam", "term", "value"])
    df["term"] = pd.Categorical(df["term"], ["generalization error", "bias$^2$", "variance"])
    best = df[df.term == "generalization error"].sort_values("value").iloc[0]
    p = (ggplot(df, aes("lam", "value", linetype="term", color="term")) + geom_line(size=0.8)
         + geom_hline(yintercept=sigma ** 2, color="grey", linetype="dotted")
         + annotate("text", x=lams[0], y=sigma ** 2, label="$\\sigma^2$ (noise)", ha="left",
                    va="bottom", size=8, color="grey")
         + geom_vline(xintercept=best.lam, color="grey", linetype="dashed")
         + scale_x_log10() + scale_color_manual(values=[PAL[4], PAL[1], PAL[0]])
         + scale_linetype_manual(values=["solid", "dashed", "dashdot"])
         + labs(x="regularization strength $\\lambda$", y="expected squared error")
         + THEME + theme(legend_position="top"))
    save(p, "bias_variance.pdf", 4.5, 3.2)


def fig_cv_curve():
    """5-fold cross-validation estimate of ridge prediction error on one simulated dataset."""
    rng = np.random.default_rng(5)
    X, y, _ = sim_highdim(rng, n=100, p=60)
    lams = np.logspace(-2, 3, 40)
    folds = rng.permutation(np.arange(len(y)) % 5)
    err = np.zeros((5, len(lams)))
    for k in range(5):
        tr, va = folds != k, folds == k
        for j, lam in enumerate(lams):
            b = ridge(X[tr], y[tr], lam)
            err[k, j] = np.mean((y[va] - X[va] @ b) ** 2)
    m, se = err.mean(0), err.std(0, ddof=1) / np.sqrt(5)
    df = pd.DataFrame({"lam": lams, "m": m, "lo": m - se, "hi": m + se})
    jbest = np.argmin(m)
    p = (ggplot(df, aes("lam", "m")) + geom_ribbon(aes(ymin="lo", ymax="hi"), fill=PAL[0], alpha=0.2)
         + geom_line(color=PAL[0], size=0.8) + geom_point(color=PAL[0], size=1)
         + geom_vline(xintercept=lams[jbest], linetype="dashed", color="grey")
         + annotate("text", x=lams[2], y=df.hi[:jbest].max() + 0.5, label="overfitting\n(overly complex model)",
                    ha="left", va="bottom", size=8)
         + annotate("text", x=lams[-5], y=m.max(), label="underfitting\n(overly simple model)", ha="right", va="bottom", size=8)
         + annotate("text", x=lams[jbest], y=m.min(), label=" optimal $\\lambda$", ha="left", va="top", size=8)
         + scale_x_log10() + labs(x="regularization strength $\\lambda$", y="cross-validation MSE")
         + THEME)
    save(p, "cv_curve.pdf", 4.5, 3.0)


# ---------------------------------------------------------------- sparsity

def fig_penalties():
    b = np.linspace(-2, 2, 401)
    df = pd.concat([pd.DataFrame({"beta": b, "pen": 0.5 * b ** 2, "panel": "L2 (ridge): $\\frac{1}{2}\\beta_j^2$"}),
                    pd.DataFrame({"beta": b, "pen": np.abs(b), "panel": "L1 (lasso): $|\\beta_j|$"})])
    df["panel"] = pd.Categorical(df["panel"], df["panel"].unique())
    p = (ggplot(df, aes("beta", "pen")) + geom_line(color=PAL[0], size=0.8)
         + facet_wrap("~panel") + labs(x="$\\beta_j$", y="penalty") + THEME)
    save(p, "penalties.pdf", 5.0, 2.4)


def fig_1d_objectives():
    """Penalized 1D objectives 0.5*(beta-b)^2 + penalty for several lambda, with their minimizers."""
    bols = 1.0
    b = np.linspace(-0.6, 1.6, 500)
    rows, mins = [], []
    for lam in [0, 0.5, 1, 2]:
        for name, pen, bhat in [("ridge", 0.5 * lam * b ** 2, bols / (1 + lam)),
                                ("lasso", lam * np.abs(b), np.sign(bols) * max(abs(bols) - lam, 0))]:
            f = 0.5 * (b - bols) ** 2 + pen
            rows.append(pd.DataFrame({"beta": b, "f": f, "lam": f"$\\lambda={lam:g}$", "panel": name}))
            fhat = 0.5 * (bhat - bols) ** 2 + (0.5 * lam * bhat ** 2 if name == "ridge" else lam * abs(bhat))
            mins.append((bhat, fhat, f"$\\lambda={lam:g}$", name))
    df = pd.concat(rows)
    md = pd.DataFrame(mins, columns=["beta", "f", "lam", "panel"])
    for d in (df, md):
        d["panel"] = pd.Categorical(d["panel"], ["ridge", "lasso"])
    p = (ggplot(df, aes("beta", "f", color="lam")) + geom_line(size=0.7)
         + geom_point(md, size=2) + geom_vline(xintercept=0, color="grey")
         + facet_wrap("~panel") + scale_color_manual(values=PAL[:4])
         + ylim(0, 2.5) + labs(x="$\\beta$", y="$\\frac{1}{2}(\\beta-\\hat\\beta_{\\mathrm{OLS}})^2$ + penalty")
         + THEME + theme(legend_position="top"))
    save(p, "objectives_1d.pdf", 5.5, 3.0)


def fig_shrinkage():
    bols = np.linspace(-3, 3, 301)
    lam = 1.0
    df = pd.concat([
        pd.DataFrame({"bols": bols, "b": bols, "est": "least squares"}),
        pd.DataFrame({"bols": bols, "b": bols / (1 + lam), "est": "ridge"}),
        pd.DataFrame({"bols": bols, "b": np.sign(bols) * np.maximum(np.abs(bols) - lam, 0), "est": "lasso"}),
    ])
    df["est"] = pd.Categorical(df["est"], ["least squares", "ridge", "lasso"])
    p = (ggplot(df, aes("bols", "b", color="est", linetype="est")) + geom_line(size=0.8)
         + scale_color_manual(values=[PAL[4], PAL[0], PAL[1]])
         + scale_linetype_manual(values=["dotted", "dashed", "solid"]) + coord_equal()
         + labs(x="$\\hat\\beta_{\\mathrm{OLS}}$", y="penalized estimate $\\hat\\beta$") + THEME)
    save(p, "shrinkage.pdf", 4.0, 3.0)


def fig_constraint_geometry():
    """Elliptical least-squares contours touching an L2 ball and an L1 ball."""
    # Q and c are chosen so the lasso solution is the corner (0, 1): at that vertex the
    # descent direction -Q(beta - c) lies in the normal cone {d : d_2 >= |d_1|}.
    Q = np.array([[1.0, 0.6], [0.6, 1.0]])
    c = np.array([1.2, 2.6])
    sq = lambda B: np.einsum("ni,ij,nj->n", B - c, Q, B - c)
    t = np.linspace(0, 2 * np.pi, 4001)
    balls = {"ridge: $\\|\\beta\\|_2\\leq\\gamma$": np.c_[np.cos(t), np.sin(t)],
             "lasso: $\\|\\beta\\|_1\\leq\\gamma$": np.c_[np.cos(t), np.sin(t)] / (np.abs(np.cos(t)) + np.abs(np.sin(t)))[:, None]}
    g = np.linspace(-2, 4, 300)
    b1, b2 = np.meshgrid(g, g)
    zz = sq(np.c_[b1.ravel(), b2.ravel()]).reshape(b1.shape)
    cont, regions, opts = [], [], []
    for name, bd in balls.items():
        v = sq(bd)
        k = np.argmin(v)
        if "lasso" in name:
            assert np.allclose(bd[k], [0, 1], atol=1e-6), bd[k]
        opts.append(pd.DataFrame({"x": [bd[k, 0]], "y": [bd[k, 1]], "panel": [name]}))
        regions.append(pd.DataFrame({"x": bd[:, 0], "y": bd[:, 1], "panel": name}))
        cc = contour_paths(b1, b2, zz, v[k] * np.array([0.15, 0.5, 1.0]))
        cc["panel"] = name
        cont.append(cc)
    cont, regions, opts = pd.concat(cont), pd.concat(regions), pd.concat(opts)
    for d in (cont, regions, opts):
        d["panel"] = pd.Categorical(d["panel"], list(balls))
    ctr = pd.DataFrame({"x": [c[0]], "y": [c[1]]})
    p = (ggplot() + geom_polygon(regions, aes("x", "y"), fill=PAL[0], alpha=0.25, color=PAL[0])
         + geom_path(cont, aes("x", "y", group="piece"), color=PAL[1], size=0.5)
         + geom_point(ctr, aes("x", "y"), shape="+", size=3, color=PAL[1])
         + geom_point(opts, aes("x", "y"), size=2.2)
         + geom_hline(yintercept=0, color="grey") + geom_vline(xintercept=0, color="grey")
         + annotate("text", x=c[0] + 0.1, y=c[1] + 0.1, label="$\\hat\\beta_{\\mathrm{OLS}}$", ha="left", size=8)
         + facet_wrap("~panel") + coord_equal(xlim=(-1.5, 3), ylim=(-1.5, 3.2))
         + labs(x="$\\beta_1$", y="$\\beta_2$") + THEME)
    save(p, "constraint_geometry.pdf", 5.5, 3.0)


def fig_sparse_priors():
    b = np.linspace(-4, 4, 2001)
    norm = lambda x, s: np.exp(-x ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)
    # unit variance for Gaussian and Laplace (scale 1/sqrt(2)); spike-and-slab with pi=0.5
    lap_b = 1 / np.sqrt(2)
    pi, eps, slab = 0.5, 0.05, np.sqrt(2.0)
    dens = {"Gaussian (ridge)": norm(b, 1),
            "Laplace (lasso)": np.exp(-np.abs(b) / lap_b) / (2 * lap_b),
            "spike-and-slab": pi * norm(b, eps) + (1 - pi) * norm(b, slab)}
    rows = []
    for k, v in dens.items():
        rows.append(pd.DataFrame({"beta": b, "value": v, "prior": k, "panel": "density"}))
        rows.append(pd.DataFrame({"beta": b, "value": np.log(v), "prior": k, "panel": "log density"}))
    df = pd.concat(rows)
    df["prior"] = pd.Categorical(df["prior"], list(dens))
    df = df[(df.panel == "log density") | (df.value < 1.2)]
    p = (ggplot(df, aes("beta", "value", color="prior", linetype="prior")) + geom_line(size=0.8)
         + facet_wrap("~panel", scales="free_y") + scale_color_manual(values=PAL[:3])
         + labs(x="$\\beta_j$", y="") + THEME + theme(legend_position="top"))
    save(p, "sparse_priors.pdf", 5.5, 3.0)


def fig_risk_vs_p():
    N, sigma2, l1sq = 1000, 1.0, 3.0
    p = np.arange(1, 501)
    df = pd.concat([pd.DataFrame({"p": p, "risk": sigma2 * p / (N - p - 1), "method": "OLS: $\\sigma^2 P/(N-P-1)$"}),
                    pd.DataFrame({"p": p, "risk": l1sq * np.sqrt(np.log(p) / N), "method": "lasso: $\\|\\beta_0\\|_1^2\\sqrt{\\log P/N}$"})])
    g = (ggplot(df, aes("p", "risk", color="method")) + geom_line(size=0.8)
         + scale_color_manual(values=[PAL[4], PAL[1]])
         + labs(x="number of covariates $P$", y="out-of-sample risk") + THEME
         + theme(legend_position="top"))
    save(g, "risk_vs_p.pdf", 4.5, 3.0)


# ---------------------------------------------------------------- optimization

def fig_gradient_descent():
    f = lambda b: 0.08 * b ** 4 - 0.5 * b ** 2 + 0.35 * b + 1.5
    df_ = lambda b: 0.32 * b ** 3 - b + 0.35
    b = np.linspace(-2.9, 2.7, 400)
    curve = pd.DataFrame({"beta": b, "L": f(b)})
    paths = []
    for start, lab in [(-2.6, "start A"), (0.8, "start B")]:
        x = [start]
        for _ in range(15):
            x.append(x[-1] - 0.3 * df_(x[-1]))
        x = np.array(x)
        paths.append(pd.DataFrame({"beta": x, "L": f(x), "run": lab}))
    paths = pd.concat(paths)
    segs = paths.groupby("run", group_keys=False).apply(
        lambda d: d.assign(beta_end=d.beta.shift(-1), L_end=d.L.shift(-1))).dropna()
    p = (ggplot(curve, aes("beta", "L")) + geom_line(size=0.8)
         + geom_segment(segs, aes(x="beta", y="L", xend="beta_end", yend="L_end", color="run"),
                        arrow=arrow(length=0.05, type="closed"), size=0.4)
         + geom_point(paths, aes(color="run"), size=1.2)
         + scale_color_manual(values=[PAL[0], PAL[1]])
         + labs(x="$\\beta$", y="loss $L(\\beta)$") + THEME + theme(legend_position="top"))
    save(p, "gradient_descent.pdf", 4.5, 3.0)


def fig_convexity():
    rows, chords, extra = [], [], []
    x = np.linspace(-2, 2, 400)
    funcs = {"convex": (lambda t: 0.4 * t ** 2 + 0.2 * t + 0.5, (-1.6, 1.7)),
             "non-convex": (lambda t: 0.25 * t ** 4 - 0.8 * t ** 2 + 0.1 * t + 1.2, (-1.3, 1.3))}
    for name, (f, (x1, x2)) in funcs.items():
        rows.append(pd.DataFrame({"x": x, "y": f(x), "panel": name}))
        chords.append(pd.DataFrame({"x": [x1, x2], "y": [f(x1), f(x2)], "panel": name}))
    rows.append(pd.DataFrame({"x": x, "y": np.abs(x), "panel": "subgradients of $|\\beta|$ at 0"}))
    for s in [-0.9, -0.5, 0, 0.5, 0.9]:
        extra.append(pd.DataFrame({"x": x, "y": s * x, "panel": "subgradients of $|\\beta|$ at 0", "s": str(s)}))
    lvl = list(funcs) + ["subgradients of $|\\beta|$ at 0"]
    df, ch, ex = pd.concat(rows), pd.concat(chords), pd.concat(extra)
    for d in (df, ch, ex):
        d["panel"] = pd.Categorical(d["panel"], lvl)
    p = (ggplot(df, aes("x", "y")) + geom_line(size=0.8)
         + geom_line(ch, color=PAL[1], linetype="dashed") + geom_point(ch, color=PAL[1], size=1.8)
         + geom_line(ex, aes(group="s"), color=PAL[3], size=0.5)
         + facet_wrap("~panel", scales="free_y") + labs(x="", y="") + THEME)
    save(p, "convexity.pdf", 6.0, 2.4)


# ---------------------------------------------------------------- data and classification

def fig_dose_response():
    """Schematic dose-response curve with IC50 and activity area."""
    logc = np.linspace(-3.2, 0.2, 300)
    ic50, hill = -1.2, 1.3
    resp = 100 / (1 + 10 ** (hill * (logc - ic50)))
    rng = np.random.default_rng(6)
    pts = np.linspace(-3, 0, 8)
    obs = 100 / (1 + 10 ** (hill * (pts - ic50))) + rng.normal(0, 3, pts.size)
    curve = pd.DataFrame({"logc": logc, "resp": resp, "top": 100.0})
    p = (ggplot(curve, aes("logc", "resp")) + geom_ribbon(aes(ymin="resp", ymax="top"), fill=PAL[2], alpha=0.3)
         + geom_line(color=PAL[0], size=0.8)
         + geom_point(pd.DataFrame({"logc": pts, "resp": obs}), size=2)
         + geom_segment(aes(x=-3.2, xend=ic50, y=50, yend=50), linetype="dashed", color="grey")
         + geom_segment(aes(x=ic50, xend=ic50, y=50, yend=0), linetype="dashed", color="grey")
         + annotate("text", x=ic50 + 0.05, y=3, label="IC$_{50}$", ha="left", size=8)
         + annotate("text", x=-0.4, y=75, label="activity area", size=8)
         + labs(x="$\\log_{10}$ drug concentration", y="growth relative to untreated (%)") + THEME)
    save(p, "dose_response.pdf", 4.5, 3.0)


def fig_sigmoid():
    g = np.linspace(-6, 6, 400)
    s = expit(g)
    df = pd.concat([pd.DataFrame({"g": g, "v": s, "panel": "$\\sigma(g)$"}),
                    pd.DataFrame({"g": g, "v": s * (1 - s), "panel": "$\\sigma'(g)=\\sigma(g)(1-\\sigma(g))$"})])
    df["panel"] = pd.Categorical(df["panel"], df["panel"].unique())
    p = (ggplot(df, aes("g", "v")) + geom_line(color=PAL[0], size=0.8)
         + geom_vline(xintercept=0, color="grey", linetype="dashed")
         + facet_wrap("~panel", scales="free_y") + labs(x="$g$", y="") + THEME)
    save(p, "sigmoid.pdf", 5.0, 2.4)


def fig_logistic_2d():
    rng = np.random.default_rng(7)
    beta = np.array([2.0, -0.8])
    X = rng.uniform(-2, 2, (60, 2))
    y = rng.random(60) < expit(X @ beta)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": np.where(y, "$y=1$", "$y=0$")})
    lines = []
    for pv in [0.1, 0.3, 0.5, 0.7, 0.9]:
        gv = np.log(pv / (1 - pv))
        x1 = np.linspace(-2, 2, 50)
        x2 = (gv - beta[0] * x1) / beta[1]
        lines.append(pd.DataFrame({"x1": x1, "x2": x2, "p": f"{pv:g}"}))
    ln = pd.concat(lines)
    ln = ln[(ln.x2 > -2.2) & (ln.x2 < 2.2)]
    lab = ln.groupby("p").apply(lambda d: d.iloc[-1]).reset_index(drop=True)
    p = (ggplot() + geom_line(ln, aes("x1", "x2", group="p", linetype="p == '0.5'"), color=PAL[2])
         + geom_text(lab, aes("x1", "x2", label="p"), size=7, color=PAL[2], nudge_y=0.12)
         + geom_point(df, aes("x1", "x2", shape="y", color="y"), size=2)
         + scale_shape_manual(values=["o", "+"]) + scale_color_manual(values=[PAL[0], PAL[1]])
         + scale_linetype_manual(values=["solid", "dashed"], guide=None)
         + coord_equal(xlim=(-2, 2), ylim=(-2.2, 2.3))
         + labs(x="$x_1$", y="$x_2$") + THEME)
    save(p, "logistic_2d.pdf", 4.0, 3.4)


def fig_roc_pr():
    rng = np.random.default_rng(8)
    n_pos, n_neg = 100, 900
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    rocs, prs = [], []
    for name, shift in [("strong", 2.0), ("weak", 0.8), ("random", 0.0)]:
        s = np.r_[rng.normal(shift, 1, n_pos), rng.normal(0, 1, n_neg)]
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        lab = f"{name} (AUROC {roc_auc_score(y, s):.2f}, AUPR {average_precision_score(y, s):.2f})"
        rocs.append(pd.DataFrame({"x": fpr, "y": tpr, "clf": lab, "panel": "ROC"}))
        prs.append(pd.DataFrame({"x": rec, "y": prec, "clf": lab, "panel": "precision-recall"}))
    df = pd.concat(rocs + prs)
    df["clf"] = pd.Categorical(df["clf"], df["clf"].unique())
    df["panel"] = pd.Categorical(df["panel"], ["ROC", "precision-recall"])
    prev = n_pos / (n_pos + n_neg)
    base = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 1, prev, prev],
                         "panel": pd.Categorical(["ROC", "ROC", "precision-recall", "precision-recall"],
                                                 ["ROC", "precision-recall"]), "g": [1, 1, 2, 2]})
    p = (ggplot(df, aes("x", "y", color="clf")) + geom_path(size=0.7)
         + geom_line(base, aes("x", "y", group="g"), inherit_aes=False, linetype="dotted", color="grey")
         + facet_wrap("~panel") + scale_color_manual(values=[PAL[0], PAL[1], PAL[4]])
         + labs(x="FPR (ROC) / recall = TPR (PR)", y="TPR (ROC) / precision (PR)")
         + THEME + theme(legend_position="bottom", legend_direction="vertical"))
    save(p, "roc_pr.pdf", 5.5, 3.6)


if __name__ == "__main__":
    for fn in [fig_linreg_scatter, fig_colinearity_contours, fig_prior_likelihood_posterior,
               fig_bias_variance, fig_cv_curve, fig_penalties, fig_1d_objectives, fig_shrinkage,
               fig_constraint_geometry, fig_sparse_priors, fig_risk_vs_p, fig_gradient_descent,
               fig_convexity, fig_dose_response, fig_sigmoid, fig_logistic_2d, fig_roc_pr]:
        fn()
        print("wrote", fn.__name__)
