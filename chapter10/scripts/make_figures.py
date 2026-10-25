"""Generate the computed figures for the splicing chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, after_stat, annotate, element_blank, element_text, facet_grid, facet_wrap,
    geom_col, geom_histogram, geom_hline, geom_line, geom_path, geom_point, geom_polygon,
    geom_raster, geom_text, geom_vline, ggplot, labs, scale_color_manual, scale_fill_cmap,
    scale_linetype_manual, theme, theme_bw, coord_equal,
)
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from scipy.stats import beta as beta_dist

import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
PAL = ["#1b6ca8", "#d1495b", "#66a182", "#edae49", "#3d3d3d", "#8e6bb3"]

THEME = theme_bw(base_size=10) + theme(
    panel_grid_minor=element_blank(),
    strip_background=element_blank(),
    strip_text=element_text(size=10),
    legend_title=element_blank(),
)


def save(p, name, w=5.0, h=3.2):
    p.save(OUT / name, width=w, height=h, verbose=False, metadata={"CreationDate": None})


# ---------------------------------------------------------------- Dirichlet

def log_dirichlet(pi, a):
    """Log Dirichlet density at the rows of pi (n x K) with parameter a (K)."""
    return gammaln(a.sum()) - gammaln(a).sum() + ((a - 1) * np.log(pi)).sum(1)


def fig_dirichlet_simplex():
    """Dirichlet densities on the 2-simplex, viewed face on, for alpha = c p."""
    settings = [(3, (1 / 3, 1 / 3, 1 / 3)), (15, (1 / 3, 1 / 3, 1 / 3)),
                (21, (1 / 3, 1 / 3, 1 / 3)), (10, (0.6, 0.2, 0.2))]
    h = np.sqrt(3) / 2
    step = 0.004
    xs, ys = np.meshgrid(np.arange(0, 1 + step, step), np.arange(0, h + step, step))
    xs, ys = xs.ravel(), ys.ravel()
    # barycentric coordinates: corner pi1=1 at (0,0), pi2=1 at (1,0), pi3=1 at (0.5,h)
    p3 = ys / h
    p2 = xs - p3 / 2
    p1 = 1 - p2 - p3
    inside = (p1 > 1e-3) & (p2 > 1e-3) & (p3 > 1e-3)
    rows, means = [], []
    for c, p in settings:
        a = c * np.array(p)
        lab = f"$c={c}$, $p=({', '.join('1/3' if abs(v - 1/3) < 1e-9 else f'{v:g}' for v in p)})$"
        dens = np.full(xs.shape, np.nan)
        dens[inside] = np.exp(log_dirichlet(np.c_[p1, p2, p3][inside], a))
        rel = dens / np.nanmax(dens)
        rows.append(pd.DataFrame({"x": xs, "y": ys, "d": rel, "panel": lab}))
        means.append(pd.DataFrame({"x": [p[1] + p[2] / 2], "y": [p[2] * h], "panel": [lab]}))
    df, mn = pd.concat(rows), pd.concat(means)
    order = list(df.panel.unique())
    for d in (df, mn):
        d["panel"] = pd.Categorical(d["panel"], order)
    tri = pd.DataFrame({"x": [0, 1, 0.5, 0], "y": [0, 0, h, 0]})
    corners = pd.DataFrame({"x": [-0.02, 1.02, 0.5], "y": [-0.07, -0.07, h + 0.06],
                            "lab": ["$\\pi_1=1$", "$\\pi_2=1$", "$\\pi_3=1$"]})
    p = (ggplot(df, aes("x", "y")) + geom_raster(aes(fill="d"))
         + geom_path(tri, aes("x", "y"), color="black", size=0.4)
         + geom_point(mn, aes("x", "y"), shape="x", size=2.5, color="black")
         + geom_text(corners, aes("x", "y", label="lab"), size=8)
         + facet_wrap("~panel", nrow=1)
         + scale_fill_cmap("magma", na_value="white", guide=None)
         + coord_equal(xlim=(-0.12, 1.12), ylim=(-0.12, h + 0.1))
         + labs(x="", y="") + THEME
         + theme(axis_text=element_blank(), axis_ticks=element_blank(),
                 panel_grid_major=element_blank(), panel_border=element_blank(),
                 strip_text=element_text(size=9)))
    save(p, "dirichlet_simplex.pdf", 7.0, 2.4)


def fig_dirichlet_draws():
    """Single draws from Dir(c p) for K=10, uniform and non-uniform means."""
    rng = np.random.default_rng(10)
    K = 10
    means = {"uniform mean": np.full(K, 1 / K)}
    pnon = rng.uniform(size=K) ** 3
    means["non-uniform mean"] = pnon / pnon.sum()
    rows = []
    for mname, pm in means.items():
        rows.append(pd.DataFrame({"k": np.arange(1, K + 1), "v": pm, "mean": mname, "row": "mean $p$"}))
        for c in [100, 30, 3]:
            rows.append(pd.DataFrame({"k": np.arange(1, K + 1), "v": rng.dirichlet(c * pm),
                                      "mean": mname, "row": f"draw, $c={c}$"}))
    df = pd.concat(rows)
    df["row"] = pd.Categorical(df["row"], ["mean $p$", "draw, $c=100$", "draw, $c=30$", "draw, $c=3$"])
    df["mean"] = pd.Categorical(df["mean"], list(means))
    p = (ggplot(df, aes("factor(k)", "v")) + geom_col(fill=PAL[0], width=0.75)
         + facet_grid("row ~ mean", scales="free_y")
         + labs(x="component $k$", y="$\\pi_k$") + THEME
         + theme(strip_text_y=element_text(size=8)))
    save(p, "dirichlet_draws.pdf", 6.0, 4.6)


# ---------------------------------------------------------------- toy isoform model

# Exons in genomic coordinates [start, end): a long first exon E1, a short version E1s,
# a middle exon E2 and a last exon E3, as in the read-ambiguity cartoon.
ISOFORMS = {
    "blue": [(0, 300), (500, 700)],
    "green": [(0, 300), (1000, 1300)],
    "red": [(0, 150), (500, 700), (1000, 1300)],
}
ISO_NAMES = list(ISOFORMS)
ISO_COLS = [PAL[0], PAL[2], PAL[1]]


def iso_coords(exons):
    return np.concatenate([np.arange(s, e) for s, e in exons])


def simulate_reads(rng, tau, n_reads, read_len):
    """Simulate reads from isoforms with molecule fractions tau; return compatibility matrix y,
    effective lengths and the read proportions theta used for simulation."""
    coords = [iso_coords(ISOFORMS[m]) for m in ISO_NAMES]
    eff = np.array([len(c) - read_len + 1 for c in coords])
    theta = tau * eff / (tau * eff).sum()
    origin = rng.choice(len(coords), size=n_reads, p=theta)
    y = np.zeros((n_reads, len(coords)), dtype=int)
    for n, m in enumerate(origin):
        s = rng.integers(eff[m])
        g = coords[m][s:s + read_len]
        for j, c in enumerate(coords):
            idx = np.searchsorted(c, g[0])
            if idx < len(c) and c[idx] == g[0] and idx + read_len <= len(c) and np.array_equal(c[idx:idx + read_len], g):
                y[n, j] = 1
    return y, eff, theta


def em(p, alpha, n_iter):
    N, M = p.shape
    theta = np.full(M, 1 / M)
    path, ll = [theta.copy()], [np.log(p @ theta).sum()]
    for _ in range(n_iter):
        phi = p * theta
        phi /= phi.sum(1, keepdims=True)
        theta = (alpha - 1 + phi.sum(0)) / (N + alpha.sum() - M)
        path.append(theta.copy())
        ll.append(np.log(p @ theta).sum())
    return np.array(path), np.array(ll)


def cavi(p, alpha, n_iter=500):
    N, M = p.shape
    a = alpha + N / M
    elbo = []
    for _ in range(n_iter):
        w = p * np.exp(digamma(a) - digamma(a.sum()))
        phi = w / w.sum(1, keepdims=True)
        a = alpha + phi.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = np.where(phi > 0, phi * (np.log(np.where(p > 0, p, 1)) - np.log(phi)), 0.0).sum()
        elbo.append(gammaln(alpha.sum()) - gammaln(a.sum()) + (gammaln(a) - gammaln(alpha)).sum() + ent)
    assert np.all(np.diff(elbo) > -1e-8), "ELBO decreased"
    return a


def gibbs(rng, p, alpha, n_samples, burn=1000):
    N, M = p.shape
    theta = np.full(M, 1 / M)
    out = []
    for it in range(n_samples + burn):
        w = p * theta
        w /= w.sum(1, keepdims=True)
        u = rng.random(N)[:, None]
        t = (np.cumsum(w, 1) < u).sum(1)
        counts = np.bincount(t, minlength=M)
        theta = rng.dirichlet(alpha + counts)
        if it >= burn:
            out.append(theta)
    return np.array(out)


def fig_em_toy():
    rng = np.random.default_rng(11)
    tau = np.array([0.5, 0.2, 0.3])
    y, eff, theta_true = simulate_reads(rng, tau, n_reads=1000, read_len=50)
    p = y / eff
    alpha = np.ones(3)
    path, ll = em(p, alpha, 40)
    assert np.all(np.diff(ll) > -1e-9)
    it = np.arange(path.shape[0])
    rows = [pd.DataFrame({"iter": it, "v": path[:, m], "iso": ISO_NAMES[m],
                          "panel": "read proportion $\\theta_m$"}) for m in range(3)]
    rows.append(pd.DataFrame({"iter": it, "v": ll, "iso": "log likelihood", "panel": "log likelihood"}))
    df = pd.concat(rows)
    df["panel"] = pd.Categorical(df["panel"], ["read proportion $\\theta_m$", "log likelihood"])
    df["iso"] = pd.Categorical(df["iso"], ISO_NAMES + ["log likelihood"])
    truth = pd.DataFrame({"v": theta_true, "iso": pd.Categorical(ISO_NAMES, ISO_NAMES + ["log likelihood"]),
                          "panel": pd.Categorical(["read proportion $\\theta_m$"] * 3, df.panel.cat.categories)})
    print("  EM toy: compat-set counts", pd.Series([tuple(r) for r in y]).value_counts().to_dict(),
          "theta_true", theta_true.round(3), "theta_EM", path[-1].round(3))
    pl = (ggplot(df, aes("iter", "v", color="iso")) + geom_line(size=0.8)
          + geom_hline(truth, aes(yintercept="v", color="iso"), linetype="dashed", size=0.5)
          + facet_wrap("~panel", scales="free_y")
          + scale_color_manual(values=ISO_COLS + [PAL[4]], breaks=ISO_NAMES)
          + labs(x="EM iteration", y="") + THEME + theme(legend_position="top"))
    save(pl, "em_toy.pdf", 6.0, 2.8)


def fig_vb_vs_gibbs():
    rng = np.random.default_rng(26)
    tau = np.array([0.5, 0.2, 0.3])
    y, eff, theta_true = simulate_reads(rng, tau, n_reads=60, read_len=100)
    p = y / eff
    alpha = np.ones(3)
    a = cavi(p, alpha)
    samples = gibbs(rng, p, alpha, 20000)
    grid = np.linspace(0.001, 0.999, 999)
    hist, lines, tru = [], [], []
    for m in range(3):
        nm = f"$\\theta_{{{m + 1}}}$ ({ISO_NAMES[m]})"
        hist.append(pd.DataFrame({"v": samples[:, m], "panel": nm}))
        lines.append(pd.DataFrame({"v": grid, "d": beta_dist.pdf(grid, a[m], a.sum() - a[m]), "panel": nm}))
        tru.append(pd.DataFrame({"v": [theta_true[m]], "panel": [nm]}))
        sd_vb = np.sqrt(a[m] * (a.sum() - a[m]) / (a.sum() ** 2 * (a.sum() + 1)))
        print(f"  {nm}: Gibbs mean {samples[:, m].mean():.3f} sd {samples[:, m].std():.3f};"
              f" CAVI mean {a[m] / a.sum():.3f} sd {sd_vb:.3f}")
    print("  Gibbs corr(theta1, theta3)", np.corrcoef(samples[:, 0], samples[:, 2])[0, 1].round(3))
    h, l, t = pd.concat(hist), pd.concat(lines), pd.concat(tru)
    order = list(h.panel.unique())
    for d in (h, l, t):
        d["panel"] = pd.Categorical(d["panel"], order)
    pl = (ggplot(h, aes("v")) + geom_histogram(aes(y=after_stat("density")), bins=60, fill="grey", alpha=0.7)
          + geom_line(l, aes("v", "d"), color=PAL[1], size=0.8)
          + geom_vline(t, aes(xintercept="v"), linetype="dashed", color="black", size=0.4)
          + facet_wrap("~panel", scales="free")
          + labs(x="read proportion", y="posterior density") + THEME)
    save(pl, "vb_vs_gibbs.pdf", 6.5, 2.4)


# ---------------------------------------------------------------- divergences

def fig_kl_directions():
    """Gaussian q minimizing the alpha-divergence D_alpha(p||q) to a bimodal p."""
    x = np.linspace(-7, 7, 4001)
    dx = x[1] - x[0]
    npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    p = 0.45 * npdf(x, -2.4, 0.45) + 0.55 * npdf(x, 1.6, 0.8)
    tiny = 1e-300

    def objective(par, a):
        m, logs = par
        q = npdf(x, m, np.exp(logs))
        if a == 0:
            return np.sum(q * (np.log(q + tiny) - np.log(p + tiny))) * dx
        if a == 1:
            return np.sum(p * (np.log(p + tiny) - np.log(q + tiny))) * dx
        return (1 - np.sum(p ** a * q ** (1 - a)) * dx) / (a * (1 - a))

    rows, curves = [], []
    labels = {0: "$\\alpha\\to 0$: KL$(q\\|p)$", 0.5: "$\\alpha=0.5$", 1: "$\\alpha=1$: KL$(p\\|q)$"}
    for a, lab in labels.items():
        best = None
        for m0 in [-2, -1, 0, 1, 2]:
            for s0 in [0.3, 1.0, 2.0]:
                r = minimize(objective, [m0, np.log(s0)], args=(a,), method="Nelder-Mead",
                             options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
                if best is None or r.fun < best.fun:
                    best = r
        m, s = best.x[0], np.exp(best.x[1])
        print(f"  alpha={a}: q = N({m:.3f}, {s:.3f}^2)")
        curves.append(pd.DataFrame({"x": x, "d": p, "dist": "$p$", "panel": lab}))
        curves.append(pd.DataFrame({"x": x, "d": npdf(x, m, s), "dist": "$q$", "panel": lab}))
    df = pd.concat(curves)
    df["panel"] = pd.Categorical(df["panel"], list(labels.values()))
    mean_p = np.sum(x * p) * dx
    sd_p = np.sqrt(np.sum((x - mean_p) ** 2 * p) * dx)
    print(f"  p mean {mean_p:.3f} sd {sd_p:.3f}")
    pl = (ggplot(df[(df.x > -5) & (df.x < 5)], aes("x", "d", color="dist")) + geom_line(size=0.8)
          + facet_wrap("~panel") + scale_color_manual(values=[PAL[0], PAL[1]])
          + labs(x="$z$", y="density") + THEME + theme(legend_position="right"))
    save(pl, "kl_directions.pdf", 6.5, 2.2)


def ellipse(S, r, n=400):
    t = np.linspace(0, 2 * np.pi, n)
    L = np.linalg.cholesky(S)
    return (L @ np.vstack([np.cos(t), np.sin(t)]) * r).T


def fig_meanfield():
    rho = 0.9
    S = np.array([[1, rho], [rho, 1]])
    Lam = np.linalg.inv(S)
    Smf = np.diag(1 / np.diag(Lam))
    assert np.allclose(np.diag(Smf), 1 - rho ** 2)
    rows = []
    for name, C in [("exact posterior", S), ("mean-field approximation", Smf)]:
        for r in [1, 2]:
            e = ellipse(C, r)
            rows.append(pd.DataFrame({"x": e[:, 0], "y": e[:, 1], "dist": name, "g": f"{name}{r}"}))
    df = pd.concat(rows)
    df["dist"] = pd.Categorical(df["dist"], ["exact posterior", "mean-field approximation"])
    pl = (ggplot(df, aes("x", "y", group="g", color="dist")) + geom_path(size=0.8)
          + geom_hline(yintercept=0, color="grey") + annotate("segment", x=0, xend=0, y=-2.5, yend=2.5, color="grey")
          + scale_color_manual(values=[PAL[5], PAL[2]]) + coord_equal(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
          + labs(x="$z_1$", y="$z_2$") + THEME + theme(legend_position="bottom", legend_direction="vertical"))
    save(pl, "meanfield.pdf", 3.4, 3.8)


def fig_psi_beta():
    grid = np.linspace(0, 1, 801)
    rows = []
    for a, b in [(7, 3), (70, 30)]:
        lab = f"$a={a}$, $b={b}$"
        rows.append(pd.DataFrame({"psi": grid, "d": beta_dist.pdf(grid, a + 1, b + 1), "counts": lab}))
        sd = beta_dist.std(a + 1, b + 1)
        print(f"  Beta({a + 1},{b + 1}) sd {sd:.3f}")
    df = pd.concat(rows)
    df["counts"] = pd.Categorical(df["counts"], df["counts"].unique())
    pl = (ggplot(df, aes("psi", "d", color="counts")) + geom_line(size=0.8)
          + scale_color_manual(values=[PAL[0], PAL[1]])
          + labs(x="$\\Psi$", y="posterior density") + THEME + theme(legend_position="top"))
    save(pl, "psi_beta.pdf", 4.0, 2.8)


if __name__ == "__main__":
    for fn in [fig_dirichlet_simplex, fig_dirichlet_draws, fig_em_toy, fig_vb_vs_gibbs,
               fig_kl_directions, fig_meanfield, fig_psi_beta]:
        fn()
        print("wrote", fn.__name__)
