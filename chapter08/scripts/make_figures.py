"""Generate the computed figures for the matrix factorization chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
All data are simulated with fixed seeds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, annotate, arrow, coord_equal, element_blank, element_text,
    facet_wrap, geom_abline, geom_area, geom_bar, geom_col, geom_hline,
    geom_line, geom_point, geom_ribbon, geom_segment, geom_text, geom_tile,
    geom_vline, ggplot, labs, scale_color_manual, scale_fill_gradient2,
    scale_fill_manual, scale_linetype_manual, scale_shape_manual, scale_size_manual,
    scale_x_continuous, scale_y_continuous, scale_y_log10, theme, theme_bw,
    coord_cartesian, guides,
)
from scipy import stats
from sklearn.decomposition import NMF, PCA, FactorAnalysis, FastICA
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.svm import SVR, NuSVR

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


def low_rank_data(rng, N, P, K, sigma=1.0, scale=None):
    """Y = X W^T + E with X ~ N(0, I_K), E ~ N(0, sigma^2 I_P). Returns centered Y and W."""
    W = rng.normal(0, 1, (P, K))
    if scale is not None:
        W = W * np.asarray(scale)[None, :]
    X = rng.normal(0, 1, (N, K))
    Y = X @ W.T + sigma * rng.normal(0, 1, (N, P))
    return Y - Y.mean(0), W


# ---------------------------------------------------------------- PCA

def fig_pca_2d():
    """2D point cloud with the principal axes, each drawn with length 2 sqrt(lambda)."""
    rng = np.random.default_rng(1)
    cov = np.array([[3.0, 1.8], [1.8, 1.6]])
    Y = rng.multivariate_normal([0, 0], cov, 300)
    Y -= Y.mean(0)
    S = Y.T @ Y / len(Y)
    lam, U = np.linalg.eigh(S)
    lam, U = lam[::-1], U[:, ::-1]
    U = U * np.sign(U[0])  # point PC1 to the right
    df = pd.DataFrame(Y, columns=["y1", "y2"])
    arr = pd.DataFrame({
        "x": [0, 0], "y": [0, 0],
        "xend": 2 * np.sqrt(lam) * U[0], "yend": 2 * np.sqrt(lam) * U[1],
        "lab": ["$\\mathbf{u}_1$", "$\\mathbf{u}_2$"],
    })
    arr["lx"] = arr.xend * 1.18
    arr["ly"] = arr.yend * 1.18
    p = (ggplot(df, aes("y1", "y2")) + geom_point(color=PAL[0], alpha=0.45, size=1)
         + geom_hline(yintercept=0, color="grey") + geom_vline(xintercept=0, color="grey")
         + geom_segment(arr, aes(x="x", y="y", xend="xend", yend="yend"), inherit_aes=False,
                        color=PAL[1], size=1, arrow=arrow(length=0.08, type="closed"))
         + geom_text(arr, aes(x="lx", y="ly", label="lab"), inherit_aes=False, size=11, color=PAL[1])
         + coord_equal() + labs(x="$y_1$", y="$y_2$") + THEME)
    save(p, "pca_2d.pdf", 4.0, 3.2)


def fig_eigen_spectrum():
    """Eigenvalues of S for K=3 signal factors plus isotropic noise, against the Marchenko-Pastur law."""
    rng = np.random.default_rng(2)
    N, P, K, sigma2 = 1000, 200, 3, 1.0
    Y, _ = low_rank_data(rng, N, P, K, np.sqrt(sigma2), scale=[0.18, 0.14, 0.11])
    lam = np.sort(np.linalg.eigvalsh(Y.T @ Y / N))[::-1]
    gamma = P / N
    edge = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lo = sigma2 * (1 - np.sqrt(gamma)) ** 2
    sig2_hat = lam[K:].mean()
    print(f"eigen_spectrum: top eigenvalues {np.round(lam[:5], 2)}, MP edge {edge:.2f}, sigma2_hat {sig2_hat:.3f}")
    df = pd.DataFrame({"k": np.arange(1, P + 1), "lam": lam,
                       "type": np.where(np.arange(P) < K, "signal", "noise")})
    p1 = (ggplot(df[df.k <= 40], aes("k", "lam", fill="type")) + geom_col(width=0.8)
          + geom_hline(yintercept=edge, linetype="dashed", color=PAL[4])
          + geom_hline(yintercept=sig2_hat, linetype="dotted", color=PAL[4])
          + annotate("text", x=40, y=edge + 0.35, label="Marchenko-Pastur edge", ha="right", size=8)
          + annotate("text", x=40, y=sig2_hat - 0.35, label="$\\hat\\sigma^2$", ha="right", size=9)
          + scale_fill_manual(values={"signal": PAL[1], "noise": PAL[0]})
          + labs(x="index $k$ (first 40 of $P=200$)", y="eigenvalue $\\lambda_k$") + THEME
          + theme(legend_position=(0.75, 0.6)))
    save(p1, "eigen_scree.pdf", 4.2, 3.0)
    # density of the bulk eigenvalues against the MP density
    x = np.linspace(lo, edge, 400)
    dens = np.sqrt((edge - x) * (x - lo)) / (2 * np.pi * sigma2 * gamma * x)
    bulk = pd.DataFrame({"lam": lam[K:]})
    curve = pd.DataFrame({"x": x, "d": dens})
    from plotnine import geom_histogram, after_stat
    p2 = (ggplot(bulk, aes("lam")) + geom_histogram(aes(y=after_stat("density")), bins=30,
                                                     fill=PAL[0], alpha=0.6)
          + geom_line(curve, aes("x", "d"), color=PAL[1], size=0.9)
          + labs(x="eigenvalue $\\lambda$ (noise eigenvalues only)", y="density") + THEME)
    save(p2, "eigen_mp.pdf", 4.2, 3.0)


def fig_fa_vs_ppca():
    """One factor loading on features 1-5; feature 6 has large independent noise.
    PPCA (spherical noise) puts its single component on the noisy feature, FA does not."""
    rng = np.random.default_rng(4)
    N, P = 500, 6
    w = np.array([1, 1, 1, 1, 1, 0.0])
    psi = np.array([0.3] * 5 + [9.0])
    x = rng.normal(size=N)
    Y = x[:, None] * w + rng.normal(size=(N, P)) * np.sqrt(psi)
    Y -= Y.mean(0)
    fa = FactorAnalysis(1).fit(Y)
    pc = PCA(1).fit(Y)
    w_fa = fa.components_.ravel()
    w_pc = (pc.components_ * np.sqrt(pc.explained_variance_ - pc.noise_variance_)).ravel()
    w_fa *= np.sign(w_fa[0]); w_pc *= np.sign(w_pc[0]) if w_pc[0] != 0 else 1
    print("fa_vs_ppca: FA", np.round(w_fa, 2), "PPCA", np.round(w_pc, 2))
    df = pd.concat([
        pd.DataFrame({"feature": np.arange(1, P + 1), "w": w, "model": "true loading"}),
        pd.DataFrame({"feature": np.arange(1, P + 1), "w": np.abs(w_fa), "model": "factor analysis"}),
        pd.DataFrame({"feature": np.arange(1, P + 1), "w": np.abs(w_pc), "model": "probabilistic PCA"}),
    ])
    df["model"] = pd.Categorical(df.model, ["true loading", "factor analysis", "probabilistic PCA"])
    p = (ggplot(df, aes("factor(feature)", "w")) + geom_col(fill=PAL[0], width=0.7)
         + facet_wrap("~model") + labs(x="feature $p$", y="$|\\hat W_{p1}|$") + THEME)
    save(p, "fa_vs_ppca.pdf", 5.5, 2.4)


def fig_gamma_ard():
    """Gamma densities for shape a < 1, a = 1 and a > 1 (rate b = 1)."""
    a_vals = [0.3, 1.0, 3.0]
    al = np.linspace(0.005, 6, 600)
    df = pd.concat([pd.DataFrame({"alpha": al, "d": stats.gamma.pdf(al, a, scale=1.0),
                                  "shape": f"$a={a:g}$"}) for a in a_vals])
    df["shape"] = pd.Categorical(df["shape"], [f"$a={a:g}$" for a in a_vals])
    p = (ggplot(df, aes("alpha", "d", color="shape")) + geom_line(size=0.9)
         + coord_cartesian(ylim=(0, 1.5))
         + scale_color_manual(values=PAL[:3])
         + labs(x="precision $\\alpha$", y="$\\mathrm{Gamma}(\\alpha\\mid a, b=1)$") + THEME
         + theme(legend_position=(0.8, 0.7)))
    save(p, "gamma_ard.pdf", 4.0, 2.8)


# ---------------------------------------------------------------- ICA

def fig_ica():
    """Gaussian vs heavy-tailed priors, and data generated from each through the same mixing matrix."""
    # densities with unit variance
    z = np.linspace(-5, 5, 800)
    nu = 3
    t_scale = np.sqrt((nu - 2) / nu)
    dens = pd.concat([
        pd.DataFrame({"z": z, "d": stats.norm.pdf(z), "prior": "Gaussian"}),
        pd.DataFrame({"z": z, "d": stats.laplace.pdf(z, scale=1 / np.sqrt(2)), "prior": "Laplace"}),
        pd.DataFrame({"z": z, "d": stats.t.pdf(z / t_scale, nu) / t_scale, "prior": "Student-$t$, $\\nu=3$"}),
    ])
    dens["prior"] = pd.Categorical(dens.prior, ["Gaussian", "Laplace", "Student-$t$, $\\nu=3$"])
    p1 = (ggplot(dens, aes("z", "d", color="prior")) + geom_line(size=0.9)
          + scale_color_manual(values=PAL[:3]) + labs(x="$x$", y="$p(x)$") + THEME
          + theme(legend_position=(0.78, 0.75)))
    save(p1, "ica_priors.pdf", 4.0, 2.8)

    rng = np.random.default_rng(5)
    N = 1500
    W = np.array([[1.0, 0.25], [0.45, 1.0]])  # columns are the true component directions
    Xg = rng.normal(size=(N, 2))
    Xs = rng.standard_t(1.5, size=(N, 2)) * 0.5  # very heavy tails for a clear picture
    rows, dirs = [], []
    for name, Xsrc in [("Gaussian factors", Xg), ("heavy-tailed factors", Xs)]:
        Y = Xsrc @ W.T + 0.05 * rng.normal(size=(N, 2))
        Y -= Y.mean(0)
        rows.append(pd.DataFrame({"y1": Y[:, 0], "y2": Y[:, 1], "panel": name}))
        L = 3.2
        for k in range(2):
            d = W[:, k] / np.linalg.norm(W[:, k])
            dirs.append(dict(x=-L * d[0], y=-L * d[1], xend=L * d[0], yend=L * d[1],
                             panel=name, method="true $\\mathbf{w}_k$"))
        pca = PCA(2).fit(Y)
        for k in range(2):
            d = pca.components_[k]
            dirs.append(dict(x=-L * d[0], y=-L * d[1], xend=L * d[0], yend=L * d[1],
                             panel=name, method="PCA"))
        if name.startswith("heavy"):
            ica = FastICA(2, whiten="unit-variance", random_state=0).fit(Y)
            A = ica.mixing_
            for k in range(2):
                d = A[:, k] / np.linalg.norm(A[:, k])
                dirs.append(dict(x=-L * d[0], y=-L * d[1], xend=L * d[0], yend=L * d[1],
                                 panel=name, method="ICA"))
            cosines = np.abs((A / np.linalg.norm(A, axis=0)).T @ (W / np.linalg.norm(W, axis=0)))
            print("ica: |cos| between ICA and true directions\n", np.round(cosines, 3))
    df = pd.concat(rows)
    dd = pd.DataFrame(dirs)
    dd["method"] = pd.Categorical(dd.method, ["true $\\mathbf{w}_k$", "PCA", "ICA"])
    p2 = (ggplot(df, aes("y1", "y2")) + geom_point(alpha=0.3, size=0.6, color="grey")
          + geom_segment(dd, aes(x="x", y="y", xend="xend", yend="yend", color="method",
                                 linetype="method", size="method"), inherit_aes=False)
          + scale_size_manual(values=[2.2, 0.8, 0.8])
          + scale_color_manual(values=["#9a9a9a", PAL[0], PAL[1]])
          + scale_linetype_manual(values=["solid", "dashed", "dashed"])
          + facet_wrap("~panel") + coord_equal(xlim=(-4, 4), ylim=(-4, 4))
          + labs(x="$y_1$", y="$y_2$") + THEME + theme(legend_position="bottom"))
    save(p2, "ica_scatter.pdf", 5.5, 3.4)


# ---------------------------------------------------------------- linear autoencoder

def fig_linear_ae():
    """Linear autoencoder trained by minibatch gradient descent converges to the PCA subspace."""
    rng = np.random.default_rng(6)
    N, P, K = 2000, 20, 3
    Y, _ = low_rank_data(rng, N, P, K, 0.5, scale=[1.5, 1.0, 0.7])
    Y /= Y.std()
    pca = PCA(K).fit(Y)
    Uk = pca.components_.T  # P x K
    mse_pca = np.mean((Y - Y @ Uk @ Uk.T) ** 2)
    # encoder A (K x P), decoder B (P x K): Yhat = Y A^T B^T
    A = 0.1 * rng.normal(size=(K, P))
    B = 0.1 * rng.normal(size=(P, K))
    lr, batch, epochs = 0.001, 50, 60
    hist = []
    for ep in range(epochs + 1):
        if ep > 0:
            perm = rng.permutation(N)
            for i in range(0, N, batch):
                Yb = Y[perm[i:i + batch]]
                Xb = Yb @ A.T
                R = Xb @ B.T - Yb  # residual, batch x P
                gB = 2 * R.T @ Xb / len(Yb) / P
                gA = 2 * (R @ B).T @ Yb / len(Yb) / P
                A -= lr * gA * P
                B -= lr * gB * P
        mse = np.mean((Y - Y @ A.T @ B.T) ** 2)
        Q, _ = np.linalg.qr(B)
        s = np.linalg.svd(Uk.T @ Q, compute_uv=False)
        angle = np.degrees(np.arccos(np.clip(s.min(), -1, 1)))
        hist.append(dict(epoch=ep, mse=mse, angle=angle))
    h = pd.DataFrame(hist)
    print(f"linear_ae: final mse {h.mse.iloc[-1]:.4f} vs PCA {mse_pca:.4f}; final angle {h.angle.iloc[-1]:.2f} deg")
    d1 = h.assign(value=h.mse, panel="reconstruction error")
    d2 = h.assign(value=h.angle, panel="angle to PCA subspace (degrees)")
    df = pd.concat([d1, d2])
    order = ["reconstruction error", "angle to PCA subspace (degrees)"]
    df["panel"] = pd.Categorical(df.panel, order)
    ref = pd.DataFrame({"panel": ["reconstruction error", "angle to PCA subspace (degrees)"],
                        "y": [mse_pca, 0.0]})
    ref["panel"] = pd.Categorical(ref.panel, order)
    p = (ggplot(df, aes("epoch", "value")) + geom_line(color=PAL[0], size=0.9)
         + geom_hline(ref, aes(yintercept="y"), linetype="dashed", color=PAL[1])
         + facet_wrap("~panel", scales="free_y") + labs(x="epoch", y="") + THEME
         + theme(subplots_adjust={"wspace": 0.3}))
    save(p, "linear_ae.pdf", 6.0, 2.6)


# ---------------------------------------------------------------- NMF

def fig_nmf_vs_pca():
    """Non-negative gene programs: NMF recovers parts, PCA loadings mix signs."""
    rng = np.random.default_rng(8)
    P, K, N = 30, 3, 300
    Wt = np.zeros((P, K))
    for k in range(K):  # each program uses a block of genes, with overlap
        Wt[k * 9:k * 9 + 12, k] = rng.uniform(1, 3, 12)
    Xt = rng.gamma(0.5, 1.0, (N, K))
    Y = Xt @ Wt.T + 0.05 * rng.gamma(1, 1, (N, P))
    nmf = NMF(K, init="nndsvda", max_iter=2000, random_state=0).fit(Y)
    Wn = nmf.components_.T
    pca = PCA(K).fit(Y)
    Wp = pca.components_.T * np.sqrt(pca.explained_variance_)
    # order NMF components to match truth
    from scipy.optimize import linear_sum_assignment
    C = np.corrcoef(np.c_[Wt, Wn].T)[:K, K:]
    _, perm = linear_sum_assignment(-C)
    Wn = Wn[:, perm]
    def norm(M):
        return M / np.abs(M).max(0, keepdims=True)
    rows = []
    for name, M in [("true $\\mathbf{W}$", Wt), ("NMF", Wn), ("PCA", Wp)]:
        M = norm(M)
        for k in range(K):
            rows.append(pd.DataFrame({"gene": np.arange(1, P + 1), "comp": k + 1, "w": M[:, k], "model": name}))
    df = pd.concat(rows)
    df["model"] = pd.Categorical(df.model, ["true $\\mathbf{W}$", "NMF", "PCA"])
    print("nmf: min PCA loading", Wp.min().round(2), "min NMF", Wn.min().round(3))
    p = (ggplot(df, aes("factor(comp)", "gene", fill="w")) + geom_tile()
         + scale_fill_gradient2(low=PAL[0], mid="white", high=PAL[1], midpoint=0, limits=(-1, 1))
         + facet_wrap("~model") + scale_y_continuous(trans="reverse")
         + labs(x="factor $k$", y="gene $p$", fill="") + THEME
         + theme(legend_title=element_blank()))
    save(p, "nmf_vs_pca.pdf", 5.0, 3.4)


# ---------------------------------------------------------------- SVR and deconvolution

def fig_svr_losses():
    e = np.linspace(-3, 3, 601)
    eps = 1.0
    df = pd.concat([
        pd.DataFrame({"e": e, "loss": e ** 2, "loss_type": "squared (L2)"}),
        pd.DataFrame({"e": e, "loss": np.abs(e), "loss_type": "absolute (L1)"}),
        pd.DataFrame({"e": e, "loss": np.maximum(0, np.abs(e) - eps),
                      "loss_type": "$\\epsilon$-insensitive, $\\epsilon=1$"}),
    ])
    df["loss_type"] = pd.Categorical(df.loss_type, ["squared (L2)", "absolute (L1)",
                                                    "$\\epsilon$-insensitive, $\\epsilon=1$"])
    p = (ggplot(df, aes("e", "loss", color="loss_type", linetype="loss_type")) + geom_line(size=0.9)
         + scale_color_manual(values=PAL[:3]) + scale_linetype_manual(values=["dashed", "dotted", "solid"])
         + coord_cartesian(ylim=(0, 4)) + labs(x="residual $r$", y="loss") + THEME
         + theme(legend_position=(0.5, 0.82)))
    save(p, "svr_losses.pdf", 4.2, 3.0)


def fig_svr_fit():
    """Linear epsilon-SVR: the tube, the support vectors, and robustness to outliers compared with OLS."""
    rng = np.random.default_rng(9)
    n = 40
    x = rng.uniform(-2, 2, n)
    y = 0.8 * x + rng.normal(0, 0.3, n)
    out_idx = np.argsort(x)[-4:]
    y[out_idx] -= rng.uniform(3, 4.5, 4)  # a few gross outliers
    X = x[:, None]
    eps = 0.3
    svr = SVR(kernel="linear", C=1.0, epsilon=eps).fit(X, y)
    ols = LinearRegression().fit(X, y)
    g = np.linspace(-2.1, 2.1, 100)
    fs = svr.predict(g[:, None])
    fo = ols.predict(g[:, None])
    sv = np.zeros(n, bool); sv[svr.support_] = True
    print(f"svr_fit: slope SVR {svr.coef_.ravel()[0]:.2f}, OLS {ols.coef_[0]:.2f}, #SV {sv.sum()}/{n}")
    pts = pd.DataFrame({"x": x, "y": y, "sv": np.where(sv, "support vector", "inside tube")})
    lines = pd.concat([pd.DataFrame({"x": g, "y": fs, "fit": "SVR"}),
                       pd.DataFrame({"x": g, "y": fo, "fit": "least squares"})])
    tube = pd.DataFrame({"x": g, "lo": fs - eps, "hi": fs + eps})
    p = (ggplot() + geom_ribbon(tube, aes("x", ymin="lo", ymax="hi"), fill=PAL[0], alpha=0.2)
         + geom_line(lines, aes("x", "y", linetype="fit", color="fit"), size=0.9)
         + geom_point(pts, aes("x", "y", shape="sv"), size=1.8)
         + scale_shape_manual(values={"support vector": "o", "inside tube": "."})
         + scale_color_manual(values={"SVR": PAL[0], "least squares": PAL[1]})
         + scale_linetype_manual(values={"SVR": "solid", "least squares": "dashed"})
         + labs(x="$x$", y="$y$") + THEME)
    save(p, "svr_fit.pdf", 4.8, 3.2)


def fig_deconvolution():
    """Simulated bulk deconvolution with an unmodelled, heavy-tailed tumour component."""
    rng = np.random.default_rng(7)
    P, K, N = 400, 5, 40
    W = np.exp(rng.normal(0, 1, (P, K)))
    for k in range(K):  # marker genes
        idx = rng.choice(P, 40, replace=False)
        W[idx, k] *= 8
    Xt = rng.dirichlet(np.ones(K), N)
    est = {"least squares": [], "least absolute deviation": [], "$\\nu$-SVR": []}
    Ws = (W - W.mean()) / W.std()
    for n in range(N):
        f = rng.uniform(0.3, 0.7)  # immune fraction of the sample
        tum = np.exp(rng.normal(0, 1, P))
        hi = rng.choice(P, 40, replace=False)
        tum[hi] *= rng.uniform(10, 40, 40)
        y = f * W @ Xt[n] + (1 - f) * tum * W.mean() + rng.normal(0, 0.1, P)
        def norm(b):
            b = np.clip(b, 0, None)
            return b / b.sum()
        est["least squares"].append(norm(LinearRegression().fit(W, y).coef_))
        est["least absolute deviation"].append(
            norm(QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(W, y).coef_))
        ys = (y - y.mean()) / y.std()
        best = None
        for nu in [0.25, 0.5, 0.75]:  # CIBERSORT-style choice of nu by reconstruction RMSE
            m = NuSVR(kernel="linear", nu=nu, C=1.0).fit(Ws, ys)
            r = np.sqrt(np.mean((m.predict(Ws) - ys) ** 2))
            if best is None or r < best[0]:
                best = (r, m.coef_.ravel())
        est["$\\nu$-SVR"].append(norm(best[1]))
    rows = []
    for name, v in est.items():
        v = np.array(v)
        rmse = np.sqrt(np.mean((v - Xt) ** 2))
        print(f"deconvolution: {name} RMSE {rmse:.3f}")
        rows.append(pd.DataFrame({"true": Xt.ravel(), "est": v.ravel(),
                                  "cell": np.tile(np.arange(1, K + 1), N).astype(str),
                                  "method": f"{name}\nRMSE = {rmse:.3f}"}))
    df = pd.concat(rows)
    df["method"] = pd.Categorical(df.method, df.method.unique())
    p = (ggplot(df, aes("true", "est", color="cell")) + geom_abline(color="grey", linetype="dashed")
         + geom_point(size=1, alpha=0.8) + facet_wrap("~method")
         + scale_color_manual(values=PAL) + coord_equal(xlim=(0, 1), ylim=(0, 1))
         + labs(x="true proportion $x_{nk}$", y="estimated proportion", color="cell type")
         + THEME + theme(legend_title=element_text(size=9)))
    save(p, "deconvolution_sim.pdf", 6.2, 2.9)


if __name__ == "__main__":
    fig_pca_2d()
    fig_eigen_spectrum()
    fig_fa_vs_ppca()
    fig_gamma_ard()
    fig_ica()
    fig_linear_ae()
    fig_nmf_vs_pca()
    fig_svr_losses()
    fig_svr_fit()
    fig_deconvolution()
