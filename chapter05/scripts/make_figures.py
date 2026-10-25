"""Generate the computed figures for the unsupervised learning chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
All data are simulated with fixed seeds.
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from plotnine import (
    aes, annotate, element_blank, element_text, facet_grid, facet_wrap, geom_col,
    geom_errorbar, geom_histogram, geom_line, geom_path, geom_point, geom_segment,
    geom_tile, geom_vline, ggplot, labs, scale_color_identity, scale_color_manual,
    scale_fill_manual, scale_linetype_manual, theme, theme_bw, coord_equal,
)
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
PAL = ["#1b6ca8", "#d1495b", "#66a182", "#edae49", "#3d3d3d", "#8e6bbf", "#5fb3c9", "#b5651d"]
PAL16 = PAL + ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#9edae5", "#ff9896", "#c5b0d5", "#2ca02c"]

THEME = theme_bw(base_size=10) + theme(
    panel_grid_minor=element_blank(),
    strip_background=element_blank(),
    strip_text=element_text(size=10),
    legend_title=element_blank(),
)


def save(p, name, w=5.0, h=3.2):
    p.save(OUT / name, width=w, height=h, verbose=False, metadata={"CreationDate": None})


def cat(df, col, levels):
    df[col] = pd.Categorical(df[col], levels)
    return df


# ---------------------------------------------------------------- shared algorithms

def kmeans_steps(X, mu, n_iter=20):
    """Plain K-means (Lloyd's algorithm). Returns the list of (centres, assignments, loss) per iteration."""
    hist = []
    for _ in range(n_iter):
        d = cdist(X, mu, "sqeuclidean")
        c = d.argmin(1)
        loss = d[np.arange(len(X)), c].sum()
        hist.append((mu.copy(), c, loss))
        new = np.array([X[c == k].mean(0) if np.any(c == k) else mu[k] for k in range(len(mu))])
        if np.allclose(new, mu):
            break
        mu = new
    return hist


def em_spherical(X, mu, sigma2, n_iter=100):
    """EM for a mixture of K spherical Gaussians with shared, fixed variance and equal weights."""
    for _ in range(n_iter):
        logr = -0.5 * cdist(X, mu, "sqeuclidean") / sigma2
        logr -= logr.max(1, keepdims=True)
        r = np.exp(logr)
        r /= r.sum(1, keepdims=True)
        mu = (r.T @ X) / r.sum(0)[:, None]
    return mu, r


def kmeanspp_init(X, K, rng):
    mu = [X[rng.integers(len(X))]]
    for _ in range(K - 1):
        d = cdist(X, np.array(mu), "sqeuclidean").min(1)
        mu.append(X[rng.choice(len(X), p=d / d.sum())])
    return np.array(mu)


def spectral_embedding(X, s, K):
    """Unnormalized spectral clustering embedding, as on the slides: L = D - A, first K eigenvectors."""
    A = np.exp(-0.5 * cdist(X, X, "sqeuclidean") / s ** 2)
    np.fill_diagonal(A, 0)
    L = np.diag(A.sum(1)) - A
    evals, U = np.linalg.eigh(L)
    return evals, U[:, :K]


def spectral_cluster(X, K, n_neighbors=10, seed=0):
    """Spectral clustering on a symmetric kNN graph with the normalized Laplacian (Ng, Jordan and Weiss)."""
    A = kneighbors_graph(X, n_neighbors, include_self=False).toarray()
    A = np.maximum(A, A.T)
    d = A.sum(1)
    Lsym = np.eye(len(X)) - A / np.sqrt(np.outer(d, d))
    _, U = np.linalg.eigh(Lsym)
    V = U[:, :K]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return KMeans(K, n_init=10, random_state=seed).fit_predict(V)


def louvain(X, n_neighbors=10, resolution=1.0, seed=0):
    """Louvain community detection on a symmetrized kNN graph."""
    A = kneighbors_graph(X, n_neighbors, include_self=False)
    A = A.maximum(A.T)
    G = nx.from_scipy_sparse_array(A)
    comms = nx.community.louvain_communities(G, resolution=resolution, seed=seed)
    lab = np.zeros(len(X), int)
    for k, cset in enumerate(sorted(comms, key=len, reverse=True)):
        lab[list(cset)] = k
    return lab


def three_clusters(rng, n=(150, 120, 100)):
    centres = np.array([[2.0, 1.0], [2.5, 5.0], [6.0, 3.0]])
    X = np.vstack([rng.normal(c, 0.6, (m, 2)) for c, m in zip(centres, n)])
    z = np.repeat(np.arange(3), n)
    return X, z, centres


# ---------------------------------------------------------------- mixture model and K-means

def fig_mixture_generative():
    """Samples from the spherical Gaussian mixture, with centres and 2-sigma circles."""
    rng = np.random.default_rng(1)
    sigma = 0.6
    X, z, centres = three_clusters(rng)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": [f"$c_i={k + 1}$" for k in z]})
    t = np.linspace(0, 2 * np.pi, 200)
    circ = pd.concat([pd.DataFrame({"x": c[0] + 2 * sigma * np.cos(t), "y": c[1] + 2 * sigma * np.sin(t),
                                    "k": f"$c_i={k + 1}$"}) for k, c in enumerate(centres)])
    cen = pd.DataFrame({"x": centres[:, 0], "y": centres[:, 1],
                        "lab": [f"$\\mu_{k + 1}$" for k in range(3)]})
    p = (ggplot(df, aes("x", "y", color="k")) + geom_point(size=0.9, alpha=0.7)
         + geom_path(circ, aes(group="k"), linetype="dashed")
         + geom_point(cen, aes("x", "y"), inherit_aes=False, size=2.5)
         + annotate("text", x=cen.x + 0.25, y=cen.y + 0.25, label=cen.lab, size=10)
         + annotate("segment", x=centres[2, 0], y=centres[2, 1], xend=centres[2, 0] + 2 * sigma,
                    yend=centres[2, 1], arrow=None)
         + annotate("text", x=centres[2, 0] + sigma, y=centres[2, 1] - 0.22, label="$2\\sigma$", size=9)
         + scale_color_manual(values=PAL[:3]) + coord_equal()
         + labs(x="$x_{i1}$", y="$x_{i2}$") + THEME)
    save(p, "mixture_generative.pdf", 4.2, 3.4)


def fig_kmeans_iterations():
    """K-means iterations (assignments as Voronoi cells, centres as crosses), and a poor local optimum."""
    rng = np.random.default_rng(2)
    X, _, _ = three_clusters(rng)
    good = kmeans_steps(X, np.array([[4.0, 0.5], [5.0, 1.0], [6.0, 5.5]]))
    bad = kmeans_steps(X, np.array([[3.0, 3.0], [6.0, 2.0], [8.0, 5.0]]))
    panels = [(f"iteration {j + 1}, $L={good[j][2]:.0f}$", good[j]) for j in range(len(good) - 1)]
    panels.append((f"converged (it. {len(good)}), $L={good[-1][2]:.0f}$", good[-1]))
    panels.append((f"other start, converged, $L={bad[-1][2]:.0f}$", bad[-1]))
    gx, gy = np.meshgrid(np.linspace(-0.5, 8.5, 120), np.linspace(-1.5, 7.0, 110))
    G = np.c_[gx.ravel(), gy.ravel()]
    pts, tiles, cents = [], [], []
    for lab, (mu, c, loss) in panels:
        pts.append(pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": c.astype(str), "panel": lab}))
        gc = cdist(G, mu, "sqeuclidean").argmin(1)
        tiles.append(pd.DataFrame({"x": G[:, 0], "y": G[:, 1], "k": gc.astype(str), "panel": lab}))
        cents.append(pd.DataFrame({"x": mu[:, 0], "y": mu[:, 1], "panel": lab}))
    pts, tiles, cents = pd.concat(pts), pd.concat(tiles), pd.concat(cents)
    levels = [lab for lab, _ in panels]
    for d in (pts, tiles, cents):
        cat(d, "panel", levels)
    p = (ggplot() + geom_tile(tiles, aes("x", "y", fill="k"), alpha=0.25)
         + geom_point(pts, aes("x", "y", color="k"), size=0.6)
         + geom_point(cents, aes("x", "y"), shape="X", size=3.5, fill="white", color="black")
         + facet_wrap("~panel", ncol=2) + scale_fill_manual(values=PAL[:3], guide=None)
         + scale_color_manual(values=PAL[:3], guide=None) + coord_equal(expand=False)
         + labs(x="$x_{i1}$", y="$x_{i2}$") + THEME)
    save(p, "kmeans_iterations.pdf", 5.6, 5.0)
    return [round(h[2]) for h in good], round(bad[-1][2])


def fig_kmeanspp():
    """Random initialization can converge to a poor local optimum; K-means++ spreads the starting centres."""
    X, _ = make_blobs(200, centers=[[-0.75, -0.8], [-0.45, -0.6], [-0.2, -0.85], [-0.05, -0.12], [0.1, -0.3]],
                      cluster_std=[0.04, 0.07, 0.06, 0.05, 0.05], random_state=3)
    K = 5
    # run many random initializations and show one that gets stuck in a poor local optimum
    losses = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        hr = kmeans_steps(X, X[rng.choice(len(X), K, replace=False)], 100)
        losses.append((hr[-1][2], seed))
    losses.sort()
    best_loss = losses[0][0]
    bad_seed = next(s for l, s in losses if l > 1.3 * best_loss)
    rng = np.random.default_rng(bad_seed)
    hr = kmeans_steps(X, X[rng.choice(len(X), K, replace=False)], 100)
    rng = np.random.default_rng(0)
    hp = kmeans_steps(X, kmeanspp_init(X, K, rng), 100)
    rows, cents = [], []
    for name, h in [("random initialization", hr), ("K-means++ initialization", hp)]:
        mu, c, loss = h[-1]
        lab = f"{name}, $L={loss:.2f}$"
        rows.append(pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": c.astype(str), "panel": lab}))
        cents.append(pd.DataFrame({"x": mu[:, 0], "y": mu[:, 1], "panel": lab}))
    rows, cents = pd.concat(rows), pd.concat(cents)
    lv = list(rows.panel.unique())
    cat(rows, "panel", lv), cat(cents, "panel", lv)
    p = (ggplot(rows, aes("x", "y", color="k")) + geom_point(size=0.9, alpha=0.8)
         + geom_point(cents, aes("x", "y"), inherit_aes=False, shape="X", size=3.5, color="black")
         + facet_wrap("~panel") + scale_color_manual(values=PAL[:5], guide=None) + coord_equal()
         + labs(x="$x_{i1}$", y="$x_{i2}$") + THEME)
    save(p, "kmeanspp.pdf", 5.6, 2.9)
    return best_loss, hr[-1][2], hp[-1][2]


def fig_choose_k():
    """Within-cluster sum of squares and the gap statistic as functions of K (true K = 4)."""
    rng = np.random.default_rng(4)
    X, _ = make_blobs(300, centers=[[0, 0], [4, 0], [0, 4], [4.5, 4.5]], cluster_std=0.8, random_state=4)
    Ks = np.arange(1, 9)
    B = 20
    lo, hi = X.min(0), X.max(0)

    def logW(Y, K):
        return np.log(KMeans(K, n_init=10, random_state=0).fit(Y).inertia_)

    lw = np.array([logW(X, K) for K in Ks])
    ref = np.array([[logW(rng.uniform(lo, hi, X.shape), K) for K in Ks] for _ in range(B)])
    gap = ref.mean(0) - lw
    sk = ref.std(0) * np.sqrt(1 + 1 / B)
    khat = next(K for j, K in enumerate(Ks[:-1]) if gap[j] >= gap[j + 1] - sk[j + 1])
    df = pd.concat([
        pd.DataFrame({"K": Ks, "v": np.exp(lw), "lo": np.nan, "hi": np.nan, "panel": "within-cluster SS $L$"}),
        pd.DataFrame({"K": Ks, "v": gap, "lo": gap - sk, "hi": gap + sk, "panel": "gap statistic"})])
    cat(df, "panel", ["within-cluster SS $L$", "gap statistic"])
    p = (ggplot(df, aes("K", "v")) + geom_line(color=PAL[0]) + geom_point(color=PAL[0], size=1.5)
         + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.2, color=PAL[0], na_rm=True)
         + geom_vline(xintercept=khat, linetype="dashed", color="grey")
         + facet_wrap("~panel", scales="free_y") + labs(x="number of clusters $K$", y="") + THEME)
    save(p, "choose_k.pdf", 5.4, 2.5)
    return khat


# ---------------------------------------------------------------- EM

def fig_jensen():
    """Jensen's inequality for the concave log: log of an average >= average of logs."""
    u = np.linspace(0.15, 5, 300)
    a, b, q = 0.5, 4.0, 0.5
    m = q * a + (1 - q) * b
    curve = pd.DataFrame({"u": u, "f": np.log(u)})
    p = (ggplot(curve, aes("u", "f")) + geom_line(color=PAL[0], size=0.8)
         + annotate("segment", x=a, y=np.log(a), xend=b, yend=np.log(b), linetype="dashed", color=PAL[1])
         + annotate("point", x=[a, b], y=[np.log(a), np.log(b)], color=PAL[1], size=2)
         + annotate("point", x=m, y=np.log(m), color=PAL[0], size=2)
         + annotate("point", x=m, y=q * np.log(a) + (1 - q) * np.log(b), color=PAL[1], size=2)
         + annotate("segment", x=m, xend=m, y=q * np.log(a) + (1 - q) * np.log(b), yend=np.log(m), color="black")
         + annotate("text", x=m + 0.1, y=np.log(m) + 0.12, label="$\\log \\mathbb{E}_q[u]$", ha="right", va="bottom", size=9)
         + annotate("text", x=m + 0.1, y=q * np.log(a) + (1 - q) * np.log(b) - 0.1,
                    label="$\\mathbb{E}_q[\\log u]$", ha="left", va="top", size=9)
         + annotate("text", x=a, y=np.log(a) - 0.15, label="$u_1$", va="top", size=9)
         + annotate("text", x=b, y=np.log(b) - 0.15, label="$u_2$", va="top", size=9)
         + labs(x="$u$", y="$\\log u$") + THEME)
    save(p, "jensen.pdf", 3.8, 2.8)


def fig_em_vs_kmeans():
    """Hard K-means assignments versus EM responsibilities (colour mixes the cluster colours)."""
    rng = np.random.default_rng(5)
    n = (200, 120, 80)
    centres = np.array([[0.0, 0.0], [5.0, 9.0], [6.0, 4.0]])
    sds = [2.0, 2.5, 3.0]
    X = np.vstack([rng.normal(c, s, (m, 2)) for c, s, m in zip(centres, sds, n)])
    mu0 = X[rng.choice(len(X), 3, replace=False)]
    hk = kmeans_steps(X, mu0, 100)
    mu_k, c_k, _ = hk[-1]
    sigma2 = 2.5 ** 2
    mu_e, r = em_spherical(X, mu0, sigma2, 200)
    # align EM cluster labels to K-means labels by nearest centre
    perm = cdist(mu_k, mu_e).argmin(1)
    r, mu_e = r[:, perm], mu_e[perm]
    base = np.array([[27, 108, 168], [209, 73, 91], [102, 161, 130]]) / 255
    hexes = lambda M: ["#%02x%02x%02x" % tuple((255 * v).round().astype(int)) for v in M]
    df = pd.concat([
        pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "col": hexes(base[c_k]), "panel": "K-means (hard)"}),
        pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "col": hexes(np.clip(r @ base, 0, 1)), "panel": "EM (soft)"})])
    cen = pd.concat([pd.DataFrame({"x": mu_k[:, 0], "y": mu_k[:, 1], "panel": "K-means (hard)"}),
                     pd.DataFrame({"x": mu_e[:, 0], "y": mu_e[:, 1], "panel": "EM (soft)"})])
    for d in (df, cen):
        cat(d, "panel", ["K-means (hard)", "EM (soft)"])
    p = (ggplot(df, aes("x", "y", color="col")) + geom_point(size=1.0) + scale_color_identity(guide=None)
         + geom_point(cen, aes("x", "y"), inherit_aes=False, shape="X", size=3.5, color="black")
         + facet_wrap("~panel") + coord_equal() + labs(x="$x_{i1}$", y="$x_{i2}$") + THEME)
    save(p, "em_vs_kmeans.pdf", 5.6, 3.0)


def fig_gmm_full_cov():
    """Elongated clusters: K-means (spherical) versus a Gaussian mixture with full covariances fit by EM."""
    rng = np.random.default_rng(6)
    covs = [np.array([[4.0, 3.2], [3.2, 3.0]]), np.array([[4.0, -3.4], [-3.4, 3.2]]), np.array([[0.3, 0], [0, 0.3]])]
    means = [np.array([0.0, 0.0]), np.array([7.0, 0.0]), np.array([3.5, -2.5])]
    n = [250, 250, 100]
    X = np.vstack([rng.multivariate_normal(m, S, k) for m, S, k in zip(means, covs, n)])
    z = np.repeat(np.arange(3), n)
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    gm = GaussianMixture(3, covariance_type="full", n_init=5, random_state=0).fit(X)
    lab_k, lab_g = km.labels_, gm.predict(X)
    t1 = f"K-means (ARI {adjusted_rand_score(z, lab_k):.2f})"
    t2 = f"Gaussian mixture, full covariance (ARI {adjusted_rand_score(z, lab_g):.2f})"
    df = pd.concat([pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": lab_k.astype(str), "panel": t1}),
                    pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": lab_g.astype(str), "panel": t2})])
    t = np.linspace(0, 2 * np.pi, 200)
    ell = []
    for k in range(3):
        w, V = np.linalg.eigh(gm.covariances_[k])
        for r in (1, 2):
            e = gm.means_[k] + r * (V @ (np.sqrt(w)[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T
            ell.append(pd.DataFrame({"x": e[:, 0], "y": e[:, 1], "g": f"{k}_{r}", "panel": t2}))
    ell = pd.concat(ell)
    for d in (df, ell):
        cat(d, "panel", [t1, t2])
    p = (ggplot(df, aes("x", "y", color="k")) + geom_point(size=0.7, alpha=0.8)
         + geom_path(ell, aes("x", "y", group="g"), inherit_aes=False, color="black", size=0.4)
         + facet_wrap("~panel") + scale_color_manual(values=PAL[:3], guide=None) + coord_equal()
         + labs(x="$x_{i1}$", y="$x_{i2}$") + THEME)
    save(p, "gmm_full_cov.pdf", 6.0, 3.0)


# ---------------------------------------------------------------- hierarchical clustering

def fig_hclust():
    """Single versus Ward linkage on the same data: dendrograms and the K=3 cuts."""
    rng = np.random.default_rng(7)
    X, z, _ = three_clusters(rng, n=(40, 30, 25))
    X = np.vstack([X, rng.uniform([0, -1], [8, 7], (8, 2))])  # a few scattered points
    names = {"single": "single linkage", "ward": "Ward linkage"}
    segs, pts = [], []
    for meth, nm in names.items():
        Z = linkage(X, meth)
        d = dendrogram(Z, no_plot=True)
        h = np.array(d["dcoord"])
        h = h / h.max()
        for xs, ys in zip(d["icoord"], h):
            for j in range(3):
                segs.append((xs[j], ys[j], xs[j + 1], ys[j + 1], nm))
        lab = fcluster(Z, 3, "maxclust")
        pts.append(pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": lab.astype(str), "panel": nm,
                                 "size": [f"{(lab == k).sum()}" for k in lab]}))
    sg = pd.DataFrame(segs, columns=["x", "y", "xend", "yend", "panel"])
    pts = pd.concat(pts)
    for d in (sg, pts):
        cat(d, "panel", list(names.values()))
    p1 = (ggplot(sg) + geom_segment(aes("x", "y", xend="xend", yend="yend"), size=0.3)
          + facet_wrap("~panel") + labs(x="data points", y="merge height (scaled)") + THEME
          + theme(axis_text_x=element_blank(), axis_ticks_major_x=element_blank()))
    save(p1, "hclust_dendro.pdf", 6.0, 2.3)
    p2 = (ggplot(pts, aes("x", "y", color="k")) + geom_point(size=1.2)
          + facet_wrap("~panel") + scale_color_manual(values=PAL[:3], guide=None) + coord_equal()
          + labs(x="$x_{i1}$", y="$x_{i2}$") + THEME)
    save(p2, "hclust_scatter.pdf", 6.0, 2.8)
    sizes = {nm: sorted(np.bincount(fcluster(linkage(X, m), 3, "maxclust"))[1:], reverse=True)
             for m, nm in names.items()}
    return sizes


# ---------------------------------------------------------------- spectral and graph clustering

def fig_spectral_moons():
    """Two moons: K-means fails, the Laplacian eigenvectors separate the moons, K-means on them succeeds."""
    X, z = make_moons(600, noise=0.06, random_state=8)
    km = KMeans(2, n_init=10, random_state=0).fit_predict(X)
    evals, U = spectral_embedding(X, s=0.1, K=2)
    sc = KMeans(2, n_init=10, random_state=0).fit_predict(U)
    t1, t2 = (f"K-means, $K=2$ (ARI {adjusted_rand_score(z, km):.2f})",
              f"spectral, $K=2$ (ARI {adjusted_rand_score(z, sc):.2f})")
    df = pd.concat([pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": km.astype(str), "panel": t1}),
                    pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": sc.astype(str), "panel": t2})])
    cat(df, "panel", [t1, t2])
    p = (ggplot(df, aes("x", "y", color="k")) + geom_point(size=0.6)
         + facet_wrap("~panel") + scale_color_manual(values=PAL[:2], guide=None) + coord_equal()
         + labs(x="$x_{i1}$", y="$x_{i2}$") + THEME)
    save(p, "spectral_moons.pdf", 6.0, 2.3)
    emb = pd.DataFrame({"x1": X[:, 0], "u2": U[:, 1], "k": sc.astype(str)})
    p2 = (ggplot(emb, aes("x1", "u2", color="k")) + geom_point(size=0.8, alpha=0.6)
          + scale_color_manual(values=PAL[:2], guide=None)
          + labs(x="$x_{i1}$", y="eigenvector 2, $U_{i2}$") + THEME)
    save(p2, "spectral_embedding.pdf", 3.2, 2.3)
    return evals[:4], adjusted_rand_score(z, km), adjusted_rand_score(z, sc), U[:, 0].std()


def fig_method_comparison():
    """K-means, spectral clustering and Louvain on toy data sets with different geometry."""
    n = 600
    X1, _ = make_circles(n, factor=0.5, noise=0.05, random_state=9)
    X2, _ = make_moons(n, noise=0.06, random_state=9)
    X3, _ = make_blobs(n, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    X4, _ = make_blobs(n, random_state=170)
    X4 = X4 @ np.array([[0.6, -0.64], [-0.41, 0.85]])
    X5 = np.random.default_rng(9).uniform(0, 1, (n, 2))
    data = {"circles": (X1, 2), "moons": (X2, 2), "unequal spread": (X3, 3),
            "anisotropic": (X4, 3), "no structure": (X5, 3)}
    rows = []
    for dn, (X, K) in data.items():
        X = (X - X.mean(0)) / X.std(0)
        labs_ = {"K-means": KMeans(K, n_init=10, random_state=0).fit_predict(X),
                 "spectral": spectral_cluster(X, K),
                 "Louvain (kNN graph)": louvain(X, 30, 1.0, 0)}
        for mn, l in labs_.items():
            rows.append(pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "k": l.astype(str),
                                      "data": dn, "method": mn}))
    df = pd.concat(rows)
    cat(df, "data", list(data))
    cat(df, "method", ["K-means", "spectral", "Louvain (kNN graph)"])
    p = (ggplot(df, aes("x", "y", color="k")) + geom_point(size=0.25)
         + facet_grid("data~method") + scale_color_manual(values=PAL16, guide=None)
         + labs(x="", y="") + THEME
         + theme(axis_text=element_blank(), axis_ticks=element_blank(), panel_grid_major=element_blank(),
                 strip_text_y=element_text(size=8)))
    save(p, "method_comparison.pdf", 5.2, 7.6)
    return {d: df[(df.data == d) & (df.method == m)].k.nunique() for d in data for m in ["Louvain (kNN graph)"]}


# ---------------------------------------------------------------- significance of clusters

def sim_single_population(rng, n, P=10):
    """One population with a skewed first axis, mimicking the long tail seen in PC1 of real data."""
    Z = rng.normal(size=(n, P))
    Z[:, 0] = rng.gamma(2.0, 1.0, n) * 2.0
    scales = np.linspace(1.0, 0.3, P)
    return Z * scales


def ward_linkage(X, lab):
    """Ward linkage between two clusters: ESS(C1 u C2) - ESS(C1) - ESS(C2)."""
    ess = lambda Y: ((Y - Y.mean(0)) ** 2).sum()
    return ess(X) - ess(X[lab == 0]) - ess(X[lab == 1])


def two_split(X):
    return fcluster(linkage(X, "ward"), 2, "maxclust") - 1


def fig_overclustering():
    """Louvain on a single simulated population finds many clusters."""
    rng = np.random.default_rng(10)
    X = sim_single_population(rng, 3000)
    lab = louvain(X, 20, 1.0, 0)
    pc = (X - X.mean(0)) @ np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:2].T
    df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "k": (lab % 8).astype(str)})
    p = (ggplot(df, aes("x", "y", color="k")) + geom_point(size=0.4, alpha=0.7)
         + scale_color_manual(values=PAL, guide=None) + labs(x="PC 1", y="PC 2") + THEME)
    save(p, "overclustering.pdf", 3.6, 2.8)
    return len(np.unique(lab))


def fig_significance():
    """Parametric bootstrap test of a two-cluster split using the Ward linkage statistic."""
    rng = np.random.default_rng(11)
    n, P, B = 600, 10, 200
    one = sim_single_population(rng, n, P)
    two = np.vstack([rng.normal(0, 1, (n // 2, P)), rng.normal(0, 1, (n // 2, P))])
    two[: n // 2, 1] += 3.0
    rows, obs, pts = [], [], []
    for name, X in [("two populations", two), ("one population", one)]:
        lab = two_split(X)
        w = ward_linkage(X, lab)
        m, S = X.mean(0), np.cov(X.T)
        null = []
        for _ in range(B):
            Y = rng.multivariate_normal(m, S, n)
            null.append(ward_linkage(Y, two_split(Y)))
        null = np.array(null)
        pval = (1 + (null >= w).sum()) / (B + 1)
        lab_p = f"{name}: $w={w:.0f}$, $p={pval:.3f}$"
        rows.append(pd.DataFrame({"w": null, "panel": lab_p}))
        obs.append(pd.DataFrame({"w": [w], "panel": [lab_p]}))
        pc = (X - m) @ np.linalg.svd(X - m, full_matrices=False)[2][:2].T
        pts.append(pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "k": lab.astype(str), "panel": name}))
    rows, obs, pts = pd.concat(rows), pd.concat(obs), pd.concat(pts)
    lv = list(rows.panel.unique())
    cat(rows, "panel", lv), cat(obs, "panel", lv)
    cat(pts, "panel", ["two populations", "one population"])
    p = (ggplot(rows, aes("w")) + geom_histogram(bins=25, fill="grey", color="white")
         + geom_vline(obs, aes(xintercept="w"), color=PAL[1], size=1)
         + facet_wrap("~panel", scales="free") + labs(x="Ward linkage $w$", y="null simulations") + THEME)
    save(p, "significance_null.pdf", 6.0, 2.4)
    p2 = (ggplot(pts, aes("x", "y", color="k")) + geom_point(size=0.5, alpha=0.7)
          + facet_wrap("~panel", scales="free") + scale_color_manual(values=PAL[:2], guide=None)
          + labs(x="PC 1", y="PC 2") + THEME)
    save(p2, "significance_data.pdf", 6.0, 2.4)
    return obs


if __name__ == "__main__":
    fig_mixture_generative()
    print("kmeans iterations (good run losses, bad optimum):", fig_kmeans_iterations())
    print("kmeans++ (best random, stuck random, k++):", fig_kmeanspp())
    print("gap K:", fig_choose_k())
    fig_jensen()
    fig_em_vs_kmeans()
    fig_gmm_full_cov()
    print("hclust sizes:", fig_hclust())
    print("spectral:", fig_spectral_moons())
    print("louvain clusters per data set:", fig_method_comparison())
    print("overclustering clusters:", fig_overclustering())
    print(fig_significance())
