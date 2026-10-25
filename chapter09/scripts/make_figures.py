"""Generate the computed figures for the biological networks chapter.

Each function writes one PDF to ../figures/. Numbers quoted in the text are
written to ../figures/network_stats.tex as LaTeX macros. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from plotnine import (
    aes, coord_equal, element_blank, element_text, facet_wrap, geom_line,
    geom_point, geom_raster, geom_segment, geom_tile, geom_vline, ggplot, labs,
    scale_color_gradient, scale_color_manual, scale_fill_gradient,
    scale_fill_gradient2, scale_x_log10, scale_y_continuous, scale_y_log10,
    scale_y_reverse, theme, theme_bw,
)
from sklearn.covariance import graphical_lasso
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

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
HEAT = THEME + theme(panel_grid_major=element_blank(), axis_ticks=element_blank(),
                     axis_text=element_blank(), axis_title=element_blank())

STATS = {}


def save(p, name, w=5.0, h=3.2):
    p.save(OUT / name, width=w, height=h, verbose=False, metadata={"CreationDate": None})


def partial_corr(theta):
    d = np.sqrt(np.diag(theta))
    r = -theta / np.outer(d, d)
    np.fill_diagonal(r, 1.0)
    return r


def matrix_long(m, panel):
    p = m.shape[0]
    i, j = np.meshgrid(np.arange(1, p + 1), np.arange(1, p + 1), indexing="ij")
    return pd.DataFrame({"row": i.ravel(), "col": j.ravel(), "value": m.ravel(), "panel": panel})


# ------------------------------------------------- partial correlation, glasso

def example_precision():
    """Sparse precision for 8 genes: a hub (gene 1 -> 2, 3, 4), a chain 5-6-7, gene 8 alone."""
    p = 8
    theta = np.eye(p)
    for a, b, w in [(0, 1, 0.45), (0, 2, 0.45), (0, 3, -0.45), (4, 5, 0.5), (5, 6, 0.5)]:
        theta[a, b] = theta[b, a] = w
    assert np.all(np.linalg.eigvalsh(theta) > 0)
    return theta


def fig_partial_correlation():
    """True partial correlations vs. marginal correlation, inverse sample covariance and glasso."""
    rng = np.random.default_rng(4)
    theta = example_precision()
    sigma = np.linalg.inv(theta)
    n = 100
    X = rng.multivariate_normal(np.zeros(8), sigma, n)
    X = X - X.mean(0)
    S = X.T @ X / n
    d = np.sqrt(np.diag(S))
    corr = S / np.outer(d, d)
    _, theta_gl = graphical_lasso(S, alpha=0.1)
    panels = ["(a) true partial corr.", "(b) sample correlation",
              "(c) partial corr. from $\\hat{\\Sigma}^{-1}$", "(d) graphical lasso"]
    mats = [partial_corr(theta), corr, partial_corr(np.linalg.inv(S)), partial_corr(theta_gl)]
    df = pd.concat([matrix_long(m, nm) for m, nm in zip(mats, panels)])
    df["panel"] = pd.Categorical(df["panel"], panels)
    df.loc[df.row == df.col, "value"] = np.nan
    p = (ggplot(df, aes("col", "row", fill="value")) + geom_tile(color="white", size=0.3)
         + scale_fill_gradient2(low=PAL[0], mid="white", high=PAL[1], midpoint=0,
                                limits=(-0.75, 0.75), na_value="#bbbbbb")
         + scale_y_reverse() + facet_wrap("~panel", nrow=1) + coord_equal() + HEAT
         + theme(legend_key_height=30, legend_key_width=8))
    save(p, "partial_correlation.pdf", 7.0, 2.2)
    STATS["corrFiveSeven"] = f"{corr[4, 6]:.2f}"
    STATS["pcorFiveSeven"] = f"{partial_corr(np.linalg.inv(S))[4, 6]:.2f}"
    STATS["glassoEdges"] = str(int((np.abs(np.triu(theta_gl, 1)) > 1e-8).sum()))


def gauss_loglik(theta, S, n):
    """Gaussian log likelihood of n centered samples with sample covariance S, without the constant."""
    sign, logdet = np.linalg.slogdet(theta)
    return 0.5 * n * (logdet - np.trace(S @ theta))


def fig_glasso_path():
    """Regularization path of partial correlations and five-fold CV log likelihood."""
    rng = np.random.default_rng(7)
    p, n = 40, 30
    theta = np.eye(p)
    edges = rng.choice(p * (p - 1) // 2, 25, replace=False)
    iu = np.triu_indices(p, 1)
    for e in edges:
        a, b = iu[0][e], iu[1][e]
        theta[a, b] = theta[b, a] = rng.choice([-1, 1]) * 0.35
    # make diagonally dominant enough to be positive definite
    theta += np.eye(p) * max(0, 0.1 - np.linalg.eigvalsh(theta).min())
    sigma = np.linalg.inv(theta)
    X = rng.multivariate_normal(np.zeros(p), sigma, n)
    X = (X - X.mean(0)) / X.std(0)
    S = X.T @ X / n
    STATS["pathRank"] = str(np.linalg.matrix_rank(S))
    lams = np.exp(np.linspace(np.log(0.03), np.log(0.8), 30))
    truth = np.abs(np.triu(theta, 1)[iu]) > 0
    rows = []
    for lam in lams:
        _, t = graphical_lasso(S, alpha=lam, max_iter=500)
        r = partial_corr(t)[iu]
        rows.append(pd.DataFrame({"lam": lam, "value": r, "pair": np.arange(len(r)),
                                  "edge": np.where(truth, "true edge", "no edge")}))
    path = pd.concat(rows)
    # keep pairs that are ever non-zero so the plot is readable
    active = path.groupby("pair")["value"].apply(lambda v: np.any(np.abs(v) > 1e-6))
    path = path[path.pair.isin(active[active].index)]
    path["panel"] = "(a) partial correlations"
    # five-fold CV
    folds = np.arange(n) % 5
    rng.shuffle(folds)
    cv = []
    for lam in lams:
        ll = 0.0
        for f in range(5):
            tr, te = X[folds != f], X[folds == f]
            Str = tr.T @ tr / len(tr)
            Ste = te.T @ te / len(te)
            _, t = graphical_lasso(Str, alpha=lam, max_iter=500)
            ll += gauss_loglik(t, Ste, len(te))
        cv.append(ll / n)
    cv = pd.DataFrame({"lam": lams, "value": cv, "panel": "(b) held-out log likelihood per sample"})
    best = lams[int(np.argmax(cv.value))]
    STATS["pathBestLambda"] = f"{best:.2f}"
    _, tb = graphical_lasso(S, alpha=best, max_iter=500)
    est = np.abs(np.triu(tb, 1)[iu]) > 1e-8
    STATS["pathBestEdges"] = str(int(est.sum()))
    STATS["pathBestTrue"] = str(int((est & truth).sum()))
    p_ = (ggplot() + geom_line(path[path.edge == "no edge"], aes("lam", "value", group="pair", color="edge"), size=0.4)
          + geom_line(path[path.edge == "true edge"], aes("lam", "value", group="pair", color="edge"), size=0.6)
          + geom_line(cv, aes("lam", "value"), color=PAL[4], size=0.7)
          + geom_point(cv, aes("lam", "value"), color=PAL[4], size=1)
          + geom_vline(xintercept=best, linetype="dashed", color="grey")
          + scale_color_manual(values={"true edge": PAL[1], "no edge": "#9a9a9a"})
          + scale_x_log10() + facet_wrap("~panel", scales="free_y")
          + labs(x="$\\lambda$", y="") + THEME + theme(legend_position="bottom"))
    save(p_, "glasso_path.pdf", 6.4, 3.0)


# ---------------------------------------------------------- network properties

def fig_betweenness():
    g = nx.random_geometric_graph(150, 0.15, seed=3)
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    pos = nx.get_node_attributes(g, "pos")
    bc = nx.betweenness_centrality(g)
    nodes = pd.DataFrame({"node": list(g.nodes), "x": [pos[v][0] for v in g.nodes],
                          "y": [pos[v][1] for v in g.nodes], "bc": [bc[v] for v in g.nodes]})
    ed = pd.DataFrame([(pos[a][0], pos[a][1], pos[b][0], pos[b][1]) for a, b in g.edges],
                      columns=["x", "y", "xend", "yend"])
    p = (ggplot() + geom_segment(ed, aes("x", "y", xend="xend", yend="yend"), color="#999999", size=0.3)
         + geom_point(nodes, aes("x", "y", color="bc"), size=2.2)
         + scale_color_gradient(low="#f4e3b0", high=PAL[0], name="betweenness")
         + coord_equal() + HEAT + theme(legend_title=element_text(size=8)))
    save(p, "betweenness.pdf", 4.6, 3.6)
    M, E = g.number_of_nodes(), g.number_of_edges()
    STATS.update(btwNodes=str(M), btwEdges=str(E), btwDensity=f"{nx.density(g):.3f}",
                 btwDiameter=str(nx.diameter(g)), btwClustering=f"{nx.average_clustering(g):.2f}",
                 btwMaxBC=f"{max(bc.values()):.2f}")


def fig_degree_distribution():
    """Degree distributions of an Erdos-Renyi and a preferential-attachment graph with equal mean degree."""
    M, m = 5000, 3
    ba = nx.barabasi_albert_graph(M, m, seed=1)
    er = nx.gnp_random_graph(M, 2 * m / (M - 1), seed=1)
    rows = []
    for nm, g in [("equal edge probability", er), ("preferential attachment", ba)]:
        deg = np.array([d for _, d in g.degree()])
        k, c = np.unique(deg[deg > 0], return_counts=True)
        rows.append(pd.DataFrame({"degree": k, "freq": c / M, "model": nm}))
        STATS["maxDegER" if g is er else "maxDegBA"] = str(deg.max())
    df = pd.concat(rows)
    p = (ggplot(df, aes("degree", "freq", color="model")) + geom_point(size=1.3, alpha=0.8)
         + scale_x_log10() + scale_y_log10()
         + scale_color_manual(values=[PAL[0], PAL[1]])
         + labs(x="degree $k$", y="fraction of nodes") + THEME
         + theme(legend_position=(0.3, 0.2)))
    save(p, "degree_distribution.pdf", 4.4, 3.0)


# --------------------------------------------------------- stochastic blockmodel

def spectral_init(A, K, rng):
    """Soft assignments from k-means on the K leading eigenvectors (by |eigenvalue|) of A."""
    w, V = np.linalg.eigh(A)
    U = V[:, np.argsort(-np.abs(w))[:K]]
    z = KMeans(K, n_init=10, random_state=int(rng.integers(1e6))).fit_predict(U)
    return 0.9 * np.eye(K)[z] + 0.1 / K


def sbm_vem(A, K, rng, iters=200, q=None):
    """Mean-field variational EM for a stochastic blockmodel with an undirected adjacency matrix."""
    M = A.shape[0]
    if q is None:
        q = rng.dirichlet(np.ones(K), M)
    off = 1 - np.eye(M)
    for _ in range(iters):
        # M step
        pi = q.mean(0)
        num = q.T @ A @ q
        den = q.T @ off @ q
        eta = np.clip(num / den, 1e-6, 1 - 1e-6)
        # E step (one sweep of parallel updates, damped)
        L1, L0 = np.log(eta), np.log1p(-eta)
        logq = np.log(pi) + A @ q @ L1.T + ((off - A) @ q) @ L0.T
        logq -= logq.max(1, keepdims=True)
        qn = np.exp(logq)
        qn /= qn.sum(1, keepdims=True)
        q = 0.5 * q + 0.5 * qn
    # evidence lower bound (expected complete log likelihood + entropy)
    ll = 0.5 * np.sum((A * (q @ L1 @ q.T) + (off - A) * (q @ L0 @ q.T)))
    elbo = ll + np.sum(q * np.log(pi)) - np.sum(q * np.log(q + 1e-300))
    return q, eta, elbo


def fig_sbm():
    rng = np.random.default_rng(11)
    K, sizes = 4, [40, 30, 30, 20]
    eta = np.array([[0.40, 0.03, 0.03, 0.03],
                    [0.03, 0.35, 0.03, 0.03],
                    [0.03, 0.03, 0.04, 0.30],
                    [0.03, 0.03, 0.30, 0.04]])
    c = np.repeat(np.arange(K), sizes)
    M = len(c)
    P = eta[c][:, c]
    U = rng.uniform(size=(M, M))
    A = np.triu(U < P, 1).astype(float)
    A = A + A.T
    perm = rng.permutation(M)
    A, c = A[perm][:, perm], c[perm]
    best = None
    for r in range(10):
        rr = np.random.default_rng(100 + r)
        q0 = spectral_init(A, K, rr) if r == 0 else None
        q, eh, elbo = sbm_vem(A, K, rr, q=q0)
        if best is None or elbo > best[2]:
            best = (q, eh, elbo)
    q, eh, _ = best
    z = q.argmax(1)
    STATS["sbmARI"] = f"{adjusted_rand_score(c, z):.2f}"
    order = np.lexsort((c, z))
    def long(Am, panel):
        i, j = np.nonzero(Am)
        return pd.DataFrame({"row": i + 1, "col": j + 1, "panel": panel})
    panels = ["(a) random node order", "(b) ordered by inferred block"]
    df = pd.concat([long(A, panels[0]), long(A[order][:, order], panels[1])])
    df["panel"] = pd.Categorical(df["panel"], panels)
    bounds = np.cumsum(np.bincount(z[order], minlength=K))[:-1] + 0.5
    lines = pd.DataFrame({"b": np.tile(bounds, 1), "panel": panels[1]})
    lines["panel"] = pd.Categorical(lines["panel"], panels)
    p = (ggplot(df, aes("col", "row")) + geom_tile(fill="black", width=1, height=1)
         + geom_vline(lines, aes(xintercept="b"), color=PAL[0], size=0.4)
         + geom_segment(lines.assign(x=0.5, xend=M + 0.5), aes(x="x", xend="xend", y="b", yend="b"),
                        color=PAL[0], size=0.4)
         + scale_y_reverse() + facet_wrap("~panel") + coord_equal() + HEAT)
    save(p, "sbm_adjacency.pdf", 6.0, 3.2)
    # report estimated eta in the inferred order of blocks
    blk = [int(z[order][0])]
    for v in z[order]:
        if v not in blk:
            blk.append(int(v))
    E = eh[np.ix_(blk, blk)]
    STATS["sbmEtaHat"] = " \\\\ ".join(" & ".join(f"{v:.2f}" for v in row) for row in E)


def write_stats():
    with open(OUT / "network_stats.tex", "w") as f:
        f.write("% generated by scripts/make_figures.py\n")
        for k, v in STATS.items():
            f.write(f"\\providecommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    fig_partial_correlation()
    fig_glasso_path()
    fig_betweenness()
    fig_degree_distribution()
    fig_sbm()
    write_stats()
    for k, v in STATS.items():
        print(k, v)
