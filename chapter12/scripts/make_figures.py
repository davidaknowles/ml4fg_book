"""Generate the computed figures for the molecular QTL chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
All data are simulated with fixed seeds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, annotate, element_blank, element_text, facet_grid, facet_wrap,
    geom_abline, geom_boxplot, geom_col, geom_errorbar, geom_errorbarh,
    geom_histogram, geom_hline, geom_jitter, geom_line, geom_point,
    geom_vline, ggplot, labs, position_dodge, scale_color_cmap,
    scale_color_manual, scale_fill_manual, scale_linetype_manual,
    scale_x_continuous, scale_y_log10, theme, theme_bw,
)
from scipy import optimize, stats

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


# ---------------------------------------------------------------- helpers

def mosaic_genotypes(rng, N, M, n_founders=8, hotspot_every=40, r_hot=0.5,
                     r_bg=0.002, mut=0.003, maf_min=0.05, n_patterns=5):
    """Genotypes (N x M, values 0/1/2) with block-like LD.

    Each haplotype is a mosaic of a few founder haplotypes, switching founder at
    recombination hotspots (every `hotspot_every` SNPs) with probability r_hot and
    elsewhere with probability r_bg, plus a small rate of allele flips. Within each
    block between hotspots, every SNP's alleles across founders follow one of
    `n_patterns` founder patterns, mimicking SNPs that arose on the same branch of
    the genealogy. This gives groups of SNPs in high LD within blocks separated by
    hotspots. SNPs with minor allele frequency below maf_min are dropped, so the
    returned M can be smaller.
    """
    founders = np.empty((n_founders, M), dtype=int)
    for start in range(0, M, hotspot_every):
        pats = []
        while len(pats) < n_patterns:
            v = (rng.uniform(size=n_founders) < rng.uniform(0.15, 0.5)).astype(int)
            if 0 < v.sum() < n_founders:
                pats.append(v)
        pats = np.array(pats)
        idx = np.arange(start, min(start + hotspot_every, M))
        founders[:, idx] = pats[rng.integers(n_patterns, size=len(idx))].T
    rec = np.full(M, r_bg)
    rec[::hotspot_every] = r_hot
    haps = np.empty((2 * N, M), dtype=int)
    cur = rng.integers(n_founders, size=2 * N)
    for m in range(M):
        sw = rng.uniform(size=2 * N) < rec[m]
        cur = np.where(sw, rng.integers(n_founders, size=2 * N), cur)
        haps[:, m] = founders[cur, m]
    flip = rng.uniform(size=haps.shape) < mut
    haps = np.where(flip, 1 - haps, haps)
    G = haps[:N] + haps[N:]
    maf = G.mean(0) / 2
    maf = np.minimum(maf, 1 - maf)
    return G[:, maf >= maf_min]


def standardize(A):
    A = A - A.mean(0)
    return A / A.std(0)


def marginal_assoc(G, y):
    """Per-SNP simple linear regression of y on each column of G: beta, se, z, p."""
    N = len(y)
    g = G - G.mean(0)
    yc = y - y.mean()
    sxx = (g ** 2).sum(0)
    beta = g.T @ yc / sxx
    resid_ss = (yc ** 2).sum() - beta ** 2 * sxx
    sigma2 = resid_ss / (N - 2)
    se = np.sqrt(sigma2 / sxx)
    z = beta / se
    p = 2 * stats.t.sf(np.abs(z), N - 2)
    return beta, se, z, p


def bh_reject(p, q=0.05):
    """Benjamini-Hochberg: boolean array of rejections at FDR q."""
    m = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


# ---------------------------------------------------------------- eQTL basics

def fig_eqtl_example():
    rng = np.random.default_rng(3)
    N = 300
    g = rng.binomial(2, 0.4, N)
    y = 0.6 * g + rng.normal(0, 1, N)
    y = (y - y.mean()) / y.std()
    b1, b0 = np.polyfit(g, y, 1)
    _, _, _, p = marginal_assoc(g[:, None].astype(float), y)
    print(f"eqtl_example: slope {b1:.2f}, p = {p[0]:.1e}")
    df = pd.DataFrame({"g": g, "y": y, "gf": g.astype(str)})
    pl = (ggplot(df, aes("g", "y"))
          + geom_boxplot(aes(group="gf"), width=0.45, outlier_shape="", color="grey")
          + geom_jitter(width=0.12, height=0, color=PAL[0], size=0.9, alpha=0.7)
          + geom_abline(intercept=b0, slope=b1, color=PAL[1], size=0.8)
          + scale_x_continuous(breaks=[0, 1, 2], labels=["GG (0)", "GA (1)", "AA (2)"])
          + labs(x="genotype at SNP (copies of allele A)", y="normalized expression")
          + THEME)
    save(pl, "eqtl_example.pdf", 3.6, 2.8)


def fig_permutation_null():
    """Null distribution of the minimum p value across a cis window, by permutation."""
    rng = np.random.default_rng(4)
    N, B = 200, 4000
    G_ld = mosaic_genotypes(rng, N, 330)
    M = G_ld.shape[1]
    G_ind = rng.binomial(2, rng.uniform(0.1, 0.5, M), (N, M))
    y = rng.normal(0, 1, N)
    Yp = np.stack([rng.permutation(y) for _ in range(B)], axis=1)
    Yp = standardize(Yp)
    rows, lines = [], []
    for name, G in [("independent SNPs", G_ind), ("SNPs in LD", G_ld)]:
        Gs = standardize(G.astype(float))
        R = Gs.T @ Yp / N                      # M x B correlations
        rmax = np.abs(R).max(0)
        t = rmax * np.sqrt((N - 2) / (1 - rmax ** 2))
        pmin = 2 * stats.t.sf(t, N - 2)
        thr = np.quantile(pmin, 0.05)
        meff = 0.05 / thr
        print(f"permutation_null ({name}): M = {M}, 5% permutation threshold {thr:.2e}, "
              f"Bonferroni {0.05 / M:.2e}, effective tests {meff:.0f}")
        lab = f"{name}\n($M_{{\\mathrm{{eff}}}}\\approx{meff:.0f}$ of $M={M}$)"
        rows.append(pd.DataFrame({"x": -np.log10(pmin), "panel": lab}))
        lines.append(pd.DataFrame({"x": [-np.log10(0.05 / M), -np.log10(thr)],
                                   "threshold": ["Bonferroni", "permutation 5%"],
                                   "panel": lab}))
    df = pd.concat(rows)
    ln = pd.concat(lines)
    order = list(df.panel.unique())
    df["panel"] = pd.Categorical(df.panel, order)
    ln["panel"] = pd.Categorical(ln.panel, order)
    pl = (ggplot(df, aes("x")) + geom_histogram(bins=50, fill="lightgrey", color="grey")
          + geom_vline(ln, aes(xintercept="x", color="threshold", linetype="threshold"), size=0.8)
          + facet_wrap("~panel")
          + scale_color_manual(values=[PAL[1], PAL[0]])
          + scale_linetype_manual(values=["dashed", "solid"])
          + labs(x="$-\\log_{10}$ (minimum p value in window) under the null", y="permutations")
          + THEME + theme(legend_position="bottom"))
    save(pl, "permutation_null.pdf", 6.0, 3.0)


# ---------------------------------------------------------------- hidden confounders

def ard_factor_analysis(Y, K, n_iter=300, seed=0):
    """Factor analysis Y (N x P) ~ X W^T + E with an ARD prior w_pk ~ N(0, 1/alpha_k).

    EM for X (posterior mean and covariance), MAP updates for W, per-gene noise
    variances psi_p and alpha_k = P / ||w_k||^2 (Bishop 1999 style). Returns E[X],
    W and the prior variances 1/alpha_k.
    """
    rng = np.random.default_rng(seed)
    N, P = Y.shape
    W = rng.normal(0, 0.1, (P, K))
    psi = Y.var(0)
    alpha = np.ones(K)
    for _ in range(n_iter):
        # E step: x_n | y_n ~ N(m_n, Sx), Sx = (I + W^T Psi^-1 W)^-1
        WtPi = W.T / psi                                  # K x P
        Sx = np.linalg.inv(np.eye(K) + WtPi @ W)
        Mx = Y @ WtPi.T @ Sx                              # N x K
        Sxx = N * Sx + Mx.T @ Mx                          # sum_n E[x x^T]
        YM = Y.T @ Mx                                     # P x K, sum_n y_pn E[x_n]
        # M step, row by row: w_p = (Sxx + psi_p A)^-1 sum_n y_pn E[x_n]
        A = np.diag(alpha)
        W = np.stack([np.linalg.solve(Sxx + psi[p] * A, YM[p]) for p in range(P)])
        psi = ((Y ** 2).sum(0) - 2 * (W * YM).sum(1) + np.einsum("pk,kl,pl->p", W, Sxx, W)) / N
        psi = np.maximum(psi, 1e-3)
        alpha = np.minimum(P / np.maximum((W ** 2).sum(0), 1e-12), 1e8)
    return Mx, W, 1 / alpha


def confounded_expression(rng, N=200, P=2000, K=10, frac_egene=0.3, b_sd=0.35):
    """Expression with one cis SNP per gene, K hidden factors and noise."""
    maf = rng.uniform(0.1, 0.5, P)
    G = rng.binomial(2, maf, (N, P)).astype(float)
    is_e = rng.uniform(size=P) < frac_egene
    b = np.where(is_e, rng.normal(0, b_sd, P), 0.0)
    X = rng.normal(0, 1, (N, K))
    scale = np.linspace(1.0, 0.3, K)
    W = rng.normal(0, 1, (P, K)) * scale * 0.7
    Y = (G - G.mean(0)) * b + X @ W.T + rng.normal(0, 1, (N, P))
    return Y - Y.mean(0), G, X, is_e


def count_egenes(Yres, G, n_cov):
    N = Yres.shape[0]
    g = G - G.mean(0)
    sxx = (g ** 2).sum(0)
    beta = (g * Yres).sum(0) / sxx
    rss = (Yres ** 2).sum(0) - beta ** 2 * sxx
    df = N - 2 - n_cov
    t = beta / np.sqrt(rss / df / sxx)
    p = 2 * stats.t.sf(np.abs(t), df)
    return bh_reject(p, 0.05)


def residualize(Y, C):
    C1 = np.column_stack([np.ones(len(Y)), C])
    coef, *_ = np.linalg.lstsq(C1, Y, rcond=None)
    return Y - C1 @ coef


def fig_hidden_factors():
    rng = np.random.default_rng(5)
    Y, G, X, is_e = confounded_expression(rng)
    N, P = Y.shape
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    ks = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60]
    rows = []
    for k in ks:
        Yr = Y if k == 0 else residualize(Y, U[:, :k])
        rej = count_egenes(Yr, G, k)
        rows.append({"k": k, "n": rej.sum(), "fdp": (rej & ~is_e).sum() / max(rej.sum(), 1)})
    df = pd.DataFrame(rows)
    rej_true = count_egenes(residualize(Y, X), G, X.shape[1])
    Kmax = 30
    Ex, W, var_k = ard_factor_analysis(Y, Kmax)
    keep = var_k > 1e-3 * var_k.max()
    rej_ard = count_egenes(residualize(Y, Ex[:, keep]), G, keep.sum())
    print("hidden_factors: eGenes by number of PCs removed:",
          dict(zip(df.k, df.n)), "| true eGenes", is_e.sum())
    print(f"hidden_factors: true factors removed {rej_true.sum()}, ARD model "
          f"({keep.sum()} active of {Kmax}) {rej_ard.sum()}; max FDP over PCs {df.fdp.max():.3f}")
    ref = pd.DataFrame({"y": [rej_true.sum(), rej_ard.sum()],
                        "what": ["true factors removed", f"ARD factor model ({keep.sum()} active)"]})
    pl = (ggplot(df, aes("k", "n")) + geom_line(color=PAL[0]) + geom_point(color=PAL[0], size=1.5)
          + geom_hline(ref, aes(yintercept="y", linetype="what", color="what"))
          + scale_linetype_manual(values=["dashed", "dotted"])
          + scale_color_manual(values=[PAL[1], PAL[2]])
          + labs(x="number of expression PCs regressed out", y="eGenes at 5% FDR")
          + THEME + theme(legend_position=(0.62, 0.25), legend_background=element_blank()))
    save(pl, "hidden_factors_power.pdf", 3.8, 2.9)

    dv = pd.DataFrame({"k": np.arange(1, Kmax + 1), "v": np.sort(var_k)[::-1]})
    dv["active"] = np.where(dv.v > 1e-3 * dv.v.max(), "active", "switched off")
    pl = (ggplot(dv, aes("k", "v", fill="active")) + geom_col(width=0.8)
          + scale_y_log10() + scale_fill_manual(values=[PAL[0], "lightgrey"])
          + geom_vline(xintercept=10.5, linetype="dashed", color="grey")
          + annotate("text", x=11.5, y=dv.v.max(), label="true $K=10$", ha="left", size=8)
          + labs(x="factor $k$ (sorted)", y="prior variance $1/\\alpha_k$")
          + THEME + theme(legend_position=(0.75, 0.6), legend_background=element_blank()))
    save(pl, "ard_factors.pdf", 3.8, 2.9)


# ---------------------------------------------------------------- LD

def fig_ld_locus():
    rng = np.random.default_rng(9)
    N = 500
    G = mosaic_genotypes(rng, N, 260, n_founders=10, mut=0.02, n_patterns=6)
    M = G.shape[1]
    c = M // 2 + 7
    y = 0.3 * (G[:, c] - G[:, c].mean()) / G[:, c].std() + rng.normal(0, 1, N)
    _, _, _, p = marginal_assoc(G.astype(float), y)
    lead = int(np.argmin(p))
    r2 = np.corrcoef(G.T)[lead] ** 2
    print(f"ld_locus: M = {M}, causal {c}, lead {lead}, r2(lead, causal) = {r2[c]:.2f}, "
          f"SNPs with p < 1e-5: {(p < 1e-5).sum()}")
    df = pd.DataFrame({"pos": np.arange(M), "lp": -np.log10(p), "r2": r2})
    df = df.sort_values("r2")
    pl = (ggplot(df, aes("pos", "lp", color="r2")) + geom_point(size=1.4)
          + scale_color_cmap("viridis", limits=(0, 1))
          + annotate("point", x=c, y=-np.log10(p[c]), shape="D", size=3, fill="none", color=PAL[1])
          + annotate("text", x=c + 6, y=-np.log10(p[c]), label="causal SNP", ha="left",
                     color=PAL[1], size=8)
          + labs(x="SNP position in cis window", y="$-\\log_{10} p$", color="$r^2$ with\nlead SNP")
          + THEME + theme(legend_title=element_text(size=9)))
    save(pl, "ld_locus.pdf", 5.0, 2.8)


# ---------------------------------------------------------------- count models

def fig_betabinom():
    n, a, b = 30, 10, 7
    y = np.arange(n + 1)
    df = pd.concat([
        pd.DataFrame({"y": y, "p": stats.binom.pmf(y, n, a / (a + b)),
                      "dist": "Binomial(30, 10/17)"}),
        pd.DataFrame({"y": y, "p": stats.betabinom.pmf(y, n, a, b),
                      "dist": "Beta-binomial(30, 10, 7)"}),
    ])
    df["dist"] = pd.Categorical(df.dist, df.dist.unique())
    for d, g in df.groupby("dist", observed=True):
        m = (g.y * g.p).sum()
        v = (g.y ** 2 * g.p).sum() - m ** 2
        print(f"betabinom: {d}: mean {m:.2f}, var {v:.2f}")
    pl = (ggplot(df, aes("y", "p", fill="dist")) + geom_col(position=position_dodge(width=0.8), width=0.8)
          + scale_fill_manual(values=[PAL[0], PAL[1]])
          + labs(x="reads from reference allele $y$ (of $n=30$)", y="probability")
          + THEME + theme(legend_position="bottom"))
    save(pl, "betabinom.pdf", 3.6, 3.0)


def fig_negbin():
    mu, shape = 30, 8
    y = np.arange(0, 81)
    df = pd.concat([
        pd.DataFrame({"y": y, "p": stats.poisson.pmf(y, mu), "dist": "Poisson(30)"}),
        pd.DataFrame({"y": y, "p": stats.nbinom.pmf(y, shape, shape / (shape + mu)),
                      "dist": "NB(mean 30, shape 8)"}),
    ])
    df["dist"] = pd.Categorical(df.dist, df.dist.unique())
    print(f"negbin: NB variance {mu + mu**2 / shape:.1f} vs Poisson {mu}")
    pl = (ggplot(df, aes("y", "p", color="dist", linetype="dist")) + geom_line(size=0.8)
          + scale_color_manual(values=[PAL[4], PAL[1]])
          + scale_linetype_manual(values=["dashed", "solid"])
          + labs(x="total read count $y$", y="probability")
          + THEME + theme(legend_position="bottom"))
    save(pl, "negbin.pdf", 3.6, 3.0)


def nb_logpmf(y, mu, shape):
    return stats.nbinom.logpmf(y, shape, shape / (shape + mu))


def fig_ase_qq():
    """Calibration and power of allele-specific tests on simulated data."""
    rng = np.random.default_rng(11)
    n, J = 20, 2000
    phi, conc = 10.0, 30.0           # NB shape and beta-binomial concentration (known)
    frac_alt = 0.2
    is_alt = rng.uniform(size=J) < frac_alt
    delta = np.where(is_alt, rng.normal(0, 0.6, J), 0.0)  # log(alpha / beta)
    m = np.exp(rng.normal(np.log(50), 0.5, J))
    T = rng.uniform(0.5, 1.5, n)                            # library size factors
    p_lm, p_cht, p_bin, p_bb = (np.full(J, np.nan) for _ in range(4))
    for j in range(J):
        g = rng.binomial(2, 0.4, n)
        while g.min() == g.max() or (g == 1).sum() < 2:
            g = rng.binomial(2, 0.4, n)
        al, be = m[j] * np.exp(delta[j] / 2), m[j] * np.exp(-delta[j] / 2)
        lam = np.select([g == 2, g == 1, g == 0], [2 * al, al + be, 2 * be])
        yt = rng.negative_binomial(phi, phi / (phi + T * lam))
        na = rng.binomial(yt, 0.2)
        pi = np.where(g == 1, al / (al + be), 0.5)
        ya = stats.betabinom.rvs(na, pi * conc, (1 - pi) * conc, random_state=rng)
        # linear model on log normalized total counts
        e = np.log((yt + 1) / T)
        p_lm[j] = stats.linregress(g, e).pvalue

        # combined haplotype-style test: NB total counts + beta-binomial allelic counts
        het = g == 1

        def nll(th):
            a, b = np.exp(th)
            lam_ = np.select([g == 2, g == 1, g == 0], [2 * a, a + b, 2 * b])
            pi_ = np.where(het, a / (a + b), 0.5)
            return -(nb_logpmf(yt, T * lam_, phi).sum()
                     + stats.betabinom.logpmf(ya, na, pi_ * conc, (1 - pi_) * conc).sum())
        th0 = np.log(np.mean(yt / T) / 2 + 0.5)
        f0 = optimize.minimize_scalar(lambda t: nll([t, t]), bracket=(th0 - 1, th0 + 1)).fun
        f1 = optimize.minimize(nll, [th0, th0], method="Nelder-Mead",
                               options={"xatol": 1e-6, "fatol": 1e-8}).fun
        p_cht[j] = stats.chi2.sf(max(2 * (f0 - f1), 0), 1)

        # allelic-only tests of p = 0.5 among heterozygotes: binomial vs beta-binomial
        ya_h, na_h = ya[het], na[het]
        phat = ya_h.sum() / max(na_h.sum(), 1)
        phat = min(max(phat, 1e-6), 1 - 1e-6)
        lr = 2 * (stats.binom.logpmf(ya_h, na_h, phat).sum() - stats.binom.logpmf(ya_h, na_h, 0.5).sum())
        p_bin[j] = stats.chi2.sf(max(lr, 0), 1)

        def bb_nll(t):
            q = 1 / (1 + np.exp(-t))
            return -stats.betabinom.logpmf(ya_h, na_h, q * conc, (1 - q) * conc).sum()
        f1b = optimize.minimize_scalar(bb_nll, bounds=(-6, 6), method="bounded").fun
        p_bb[j] = stats.chi2.sf(max(2 * (bb_nll(0.0) - f1b), 0), 1)

    def qq(p, method, panel):
        p = np.sort(p)
        k = len(p)
        return pd.DataFrame({"exp": -np.log10((np.arange(1, k + 1) - 0.5) / k),
                             "obs": -np.log10(p), "method": method, "panel": panel})
    null = ~is_alt
    pa = "(a) null genes: allelic test"
    pb = "(b) all genes: eQTL test"
    df = pd.concat([qq(p_bin[null], "binomial", pa), qq(p_bb[null], "beta-binomial", pa),
                    qq(p_lm, "linear model", pb), qq(p_cht, "combined NB + beta-binomial", pb)])
    df["method"] = pd.Categorical(df.method, ["binomial", "beta-binomial", "linear model",
                                              "combined NB + beta-binomial"])
    print(f"ase_qq: null allelic tests with p<0.05: binomial {np.mean(p_bin[null] < 0.05):.2f}, "
          f"beta-binomial {np.mean(p_bb[null] < 0.05):.3f}")
    print(f"ase_qq: BH 5% discoveries of {is_alt.sum()} true: linear {bh_reject(p_lm).sum()}, "
          f"combined {bh_reject(p_cht).sum()}; null combined p<0.05 {np.mean(p_cht[null] < 0.05):.3f}")
    pl = (ggplot(df, aes("exp", "obs", color="method")) + geom_abline(linetype="dashed", color="grey")
          + geom_point(size=0.7) + facet_wrap("~panel", scales="free_y")
          + scale_color_manual(values=[PAL[3], PAL[2], PAL[0], PAL[4]])
          + labs(x="expected $-\\log_{10} p$", y="observed $-\\log_{10} p$")
          + THEME + theme(legend_position="bottom"))
    save(pl, "ase_qq.pdf", 6.0, 3.4)


# ---------------------------------------------------------------- tissue sharing

def fig_sharing_naive():
    rng = np.random.default_rng(12)
    J, n1, n2 = 3000, 500, 150
    b = rng.normal(0, 0.15, J)          # identical true effect in both tissues
    z1 = b * np.sqrt(n1) + rng.normal(size=J)
    z2 = b * np.sqrt(n2) + rng.normal(size=J)
    zt = stats.norm.isf(5e-5)           # two-sided p < 1e-4
    s1, s2 = np.abs(z1) > zt, np.abs(z2) > zt
    cat = np.select([s1 & s2, s1 & ~s2, ~s1 & s2], ["significant in both",
                    "tissue 1 only", "tissue 2 only"], "neither")
    df = pd.DataFrame({"z1": z1, "z2": z2, "cat": cat})
    df["cat"] = pd.Categorical(df.cat, ["significant in both", "tissue 1 only", "tissue 2 only", "neither"])
    any_sig = s1 | s2
    print(f"sharing_naive: of {any_sig.sum()} eQTLs significant somewhere, "
          f"{np.mean(cat[any_sig] == 'tissue 1 only'):.2f} called tissue-1 specific, "
          f"{np.mean(cat[any_sig] == 'tissue 2 only'):.3f} tissue-2 specific")
    pl = (ggplot(df, aes("z1", "z2", color="cat")) + geom_point(size=0.6, alpha=0.7)
          + geom_abline(slope=np.sqrt(n2 / n1), color="grey", linetype="dashed")
          + scale_color_manual(values=[PAL[0], PAL[1], PAL[3], "lightgrey"])
          + labs(x=f"$z$ score, tissue 1 ($n={n1}$)", y=f"$z$ score, tissue 2 ($n={n2}$)")
          + THEME)
    save(pl, "sharing_naive.pdf", 4.8, 3.0)


# ---------------------------------------------------------------- colocalization

def log_abf(beta, se, W=0.15 ** 2):
    """Wakefield's approximate Bayes factor (log) for N(0, W) prior on the effect."""
    V = se ** 2
    z2 = (beta / se) ** 2
    return 0.5 * np.log(V / (V + W)) + 0.5 * z2 * W / (V + W)


def coloc_posteriors(l1, l2, p1=1e-4, p2=1e-4, p12=1e-5):
    from scipy.special import logsumexp
    s1, s2 = logsumexp(l1), logsumexp(l2)
    s12 = logsumexp(l1 + l2)
    lh = np.array([
        0.0,
        np.log(p1) + s1,
        np.log(p2) + s2,
        np.log(p1) + np.log(p2) + s1 + s2 + np.log1p(-np.exp(s12 - s1 - s2)),
        np.log(p12) + s12,
    ])
    return np.exp(lh - logsumexp(lh))


def fig_coloc():
    rng = np.random.default_rng(21)
    N1, N2, Mraw = 500, 5000, 200
    G_all = mosaic_genotypes(rng, N1 + N2, Mraw, hotspot_every=50)
    G1, G2 = G_all[:N1].astype(float), G_all[N1:].astype(float)
    M = G_all.shape[1]
    R = np.corrcoef(G2.T)
    c1 = M // 2
    # a second causal SNP in weak LD with c1, in a neighbouring LD block, for the H3 scenario
    cand = [j for j in range(M) if 0.1 < R[c1, j] ** 2 < 0.4 and abs(j - c1) > 25]
    c2 = cand[0]
    print(f"coloc: M = {M}, c1 = {c1}, c2 = {c2}, r2 = {R[c1, c2] ** 2:.2f}")
    sd = lambda g: (g - g.mean()) / g.std()
    y1 = 0.4 * sd(G1[:, c1]) + rng.normal(size=N1)
    rows, titles = [], {}
    for scen, c in [("shared causal variant", c1), ("distinct causal variants", c2)]:
        y2 = 0.09 * sd(G2[:, c]) + rng.normal(size=N2)
        b1, se1, _, p1 = marginal_assoc(G1, (y1 - y1.mean()) / y1.std())
        b2, se2, _, p2 = marginal_assoc(G2, (y2 - y2.mean()) / y2.std())
        pp = coloc_posteriors(log_abf(b1, se1), log_abf(b2, se2))
        print(f"coloc ({scen}): PP H0..H4 = " + ", ".join(f"{x:.3f}" for x in pp))
        lab = f"{scen}\nPP(H3) = {pp[3]:.2f}, PP(H4) = {pp[4]:.2f}"
        titles[scen] = lab
        for trait, p, cc in [("eQTL ($n=500$)", p1, c1), ("GWAS ($n=5000$)", p2, c)]:
            rows.append(pd.DataFrame({"pos": np.arange(M), "lp": -np.log10(p), "trait": trait,
                                      "scen": lab, "causal": np.arange(M) == cc}))
    df = pd.concat(rows)
    df["scen"] = pd.Categorical(df.scen, list(titles.values()))
    df["trait"] = pd.Categorical(df.trait, ["eQTL ($n=500$)", "GWAS ($n=5000$)"])
    pl = (ggplot(df, aes("pos", "lp", color="trait")) + geom_point(size=0.8)
          + geom_point(df[df.causal], shape="D", size=2.6, fill="none", color="black")
          + facet_grid("trait~scen", scales="free_y")
          + scale_color_manual(values=[PAL[1], PAL[0]])
          + labs(x="SNP position", y="$-\\log_{10} p$") + THEME + theme(legend_position="none"))
    save(pl, "coloc_example.pdf", 6.0, 3.8)


# ---------------------------------------------------------------- Mendelian randomization

def fig_mr_egger():
    rng = np.random.default_rng(31)
    J, beta = 25, 0.2
    gam = rng.uniform(0.03, 0.3, J)
    alpha = rng.normal(0.02, 0.01, J)       # directional pleiotropy, independent of gam
    se_g, se_G = np.full(J, 0.01), np.full(J, 0.01)
    gh = gam + rng.normal(0, se_g)
    Gh = beta * gam + alpha + rng.normal(0, se_G)
    w = 1 / se_G ** 2
    b_ivw = np.sum(w * gh * Gh) / np.sum(w * gh ** 2)
    Xe = np.column_stack([np.ones(J), gh])
    coef = np.linalg.solve(Xe.T @ (w[:, None] * Xe), Xe.T @ (w * Gh))
    b0, b_eg = coef
    ratio = Gh / gh
    print(f"mr_egger: true beta {beta}, IVW {b_ivw:.3f}, Egger slope {b_eg:.3f}, "
          f"intercept {b0:.3f} (mean alpha {alpha.mean():.3f}); median ratio {np.median(ratio):.3f}")
    df = pd.DataFrame({"g": gh, "G": Gh, "sg": se_g, "sG": se_G})
    ln = pd.DataFrame({"a": [0, b0, 0], "b": [b_ivw, b_eg, beta],
                       "what": [f"IVW, slope {b_ivw:.2f}", f"Egger, slope {b_eg:.2f}",
                                f"true causal effect {beta}"]})
    ln["what"] = pd.Categorical(ln.what, ln.what)
    pl = (ggplot(df, aes("g", "G"))
          + geom_abline(ln, aes(intercept="a", slope="b", color="what", linetype="what"), size=0.8)
          + geom_errorbar(aes(ymin="G-1.96*sG", ymax="G+1.96*sG"), width=0, color="grey")
          + geom_errorbarh(aes(xmin="g-1.96*sg", xmax="g+1.96*sg"), height=0, color="grey")
          + geom_point(size=1.5)
          + scale_color_manual(values=[PAL[1], PAL[0], PAL[4]])
          + scale_linetype_manual(values=["solid", "solid", "dashed"])
          + labs(x="SNP effect on exposure $\\hat\\gamma_j$", y="SNP effect on outcome $\\hat\\Gamma_j$")
          + THEME + theme(legend_position="right"))
    save(pl, "mr_egger.pdf", 5.6, 3.0)


if __name__ == "__main__":
    fig_eqtl_example()
    fig_permutation_null()
    fig_hidden_factors()
    fig_ld_locus()
    fig_betabinom()
    fig_negbin()
    fig_ase_qq()
    fig_sharing_naive()
    fig_coloc()
    fig_mr_egger()
