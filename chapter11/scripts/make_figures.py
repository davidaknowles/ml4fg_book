"""Generate the computed figures for the GWAS chapter.

Each function writes one PDF to ../figures/. All data are simulated with fixed seeds.
Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, annotate, coord_equal, element_blank, element_text, facet_wrap,
    geom_abline, geom_col, geom_hline, geom_line, geom_point, geom_rect,
    geom_segment, geom_tile, geom_vline, ggplot, guide_legend, guides, labs,
    position_dodge, scale_color_manual, scale_fill_gradient, scale_fill_manual,
    scale_shape_manual, scale_x_continuous, scale_x_log10, scale_y_continuous,
    scale_y_log10, theme, theme_bw,
)
from scipy import stats
from scipy.optimize import brentq

import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
PAL = ["#1b6ca8", "#d1495b", "#66a182", "#edae49", "#3d3d3d"]
GWS = 5e-8  # genome-wide significance threshold

THEME = theme_bw(base_size=10) + theme(
    panel_grid_minor=element_blank(),
    strip_background=element_blank(),
    strip_text=element_text(size=10),
    legend_title=element_blank(),
)


def save(p, name, w=5.0, h=3.2):
    p.save(OUT / name, width=w, height=h, verbose=False, metadata={"CreationDate": None})


# ---------------------------------------------------------------- simulators

def mosaic_haplotypes(rng, n_hap, founders, switch, weights=None, err=0.005):
    """Haplotypes as mosaics of founder haplotypes.

    founders: (F, P) binary array. switch: (P-1,) probability that the copied founder
    changes between SNP j and j+1 (a recombination in the genealogy). weights: (F,)
    probabilities of copying each founder (population-specific if desired).
    err: per-site probability of flipping the copied allele (mutation).
    """
    F, P = founders.shape
    w = np.full(F, 1.0 / F) if weights is None else weights
    state = rng.choice(F, n_hap, p=w)
    H = np.empty((n_hap, P), dtype=np.int8)
    H[:, 0] = founders[state, 0]
    for j in range(1, P):
        sw = rng.random(n_hap) < switch[j - 1]
        state[sw] = rng.choice(F, sw.sum(), p=w)
        H[:, j] = founders[state, j]
    flip = rng.random(H.shape) < err
    return np.where(flip, 1 - H, H)


def coalescent_clades(rng, F):
    """Clades (sets of founders below each branch) and branch lengths of a coalescent tree."""
    lineages = [(frozenset([f]), 0.0) for f in range(F)]  # (clade, time it was born)
    clades, lengths, t = [], [], 0.0
    while len(lineages) > 1:
        k = len(lineages)
        t += rng.exponential(2.0 / (k * (k - 1)))
        i, j = sorted(rng.choice(k, 2, replace=False))
        (ci, ti), (cj, tj) = lineages[i], lineages[j]
        clades += [ci, cj]
        lengths += [t - ti, t - tj]
        lineages = [l for m, l in enumerate(lineages) if m not in (i, j)] + [(ci | cj, t)]
    return clades, np.array(lengths)


def random_founders(rng, F, P, block=None):
    """Founder haplotypes with tree-structured LD: within each block the SNPs are mutations
    on the branches of one coalescent tree; different blocks have independent trees."""
    block = np.zeros(P, int) if block is None else np.asarray(block)
    H = np.zeros((F, P), dtype=np.int8)
    for b in np.unique(block):
        clades, lengths = coalescent_clades(rng, F)
        cols = np.where(block == b)[0]
        pick = rng.choice(len(clades), len(cols), p=lengths / lengths.sum())
        for col, k in zip(cols, pick):
            H[list(clades[k]), col] = 1
    return H


def blocks_from_hotspots(P, hotspots):
    return np.searchsorted(np.asarray(hotspots) + 0.5, np.arange(P))


def switch_profile(P, bg, hot, hotspots):
    s = np.full(P - 1, bg)
    s[np.asarray(hotspots)] = hot
    return s


def genotypes_from_haps(H):
    return H[0::2] + H[1::2]


def standardize(X):
    X = X.astype(float)
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


def maf(X):
    f = X.mean(0) / 2
    return np.minimum(f, 1 - f)


def marginal_z(Xs, y):
    """Per-SNP regression z statistics for standardized genotypes Xs (N x P)."""
    N = len(y)
    ys = (y - y.mean()) / y.std()
    r = Xs.T @ ys / N
    return r * np.sqrt((N - 2) / (1 - r ** 2))


# ---------------------------------------------------------------- variation

def fig_sfs():
    """Folded site frequency spectrum under the neutral constant-size model."""
    n = 20000  # sampled chromosomes
    i = np.arange(1, n)
    xi = 1.0 / i  # expected number of sites with derived count i (up to a constant)
    p = i / n
    m = np.minimum(p, 1 - p)
    het = xi * 2 * p * (1 - p)  # expected heterozygous sites in one individual
    edges = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5001]
    labels = ["<0.1", "0.1-1", "1-5", "5-10", "10-20", "20-30", "30-40", "40-50"]
    b = np.digitize(m, edges) - 1
    rows = []
    for k, lab in enumerate(labels):
        rows.append({"bin": lab, "what": "share of SNPs", "frac": xi[b == k].sum() / xi.sum()})
        rows.append({"bin": lab, "what": "share of an individual's heterozygous sites",
                     "frac": het[b == k].sum() / het.sum()})
    df = pd.DataFrame(rows)
    df["bin"] = pd.Categorical(df["bin"], labels)
    df["what"] = pd.Categorical(df["what"], ["share of SNPs", "share of an individual's heterozygous sites"])
    p_ = (ggplot(df, aes("bin", "frac", fill="what"))
          + geom_col(position=position_dodge(width=0.8), width=0.75)
          + scale_fill_manual(values=[PAL[0], PAL[3]])
          + labs(x="minor allele frequency (%)", y="proportion")
          + THEME + theme(legend_position="top"))
    save(p_, "sfs.pdf", 5.5, 3.2)


# ---------------------------------------------------------------- GWAS basics

CHR_MB = [248, 242, 198, 190, 181, 171, 159, 145, 138, 134, 135, 133, 114, 107, 102,
          90, 83, 80, 59, 64, 47, 51]


def fig_manhattan():
    """Manhattan plot from z-scores simulated as zhat ~ N(R z, R) with AR(1) LD."""
    rng = np.random.default_rng(11)
    per_mb, rho = 8, 0.8
    rows, offset = [], 0
    n_causal = {c: 0 for c in range(1, 23)}
    for c in rng.choice(np.arange(1, 23), 30):
        n_causal[c] += 1
    for c, mb in enumerate(CHR_MB, start=1):
        P = mb * per_mb
        e = np.empty(P)
        e[0] = rng.normal()
        eta = rng.normal(size=P)
        for j in range(1, P):
            e[j] = rho * e[j - 1] + np.sqrt(1 - rho ** 2) * eta[j]
        mean = np.zeros(P)
        idx = np.arange(P)
        for _ in range(n_causal[c]):
            k = rng.integers(P)
            zc = rng.uniform(3.5, 10.0) * rng.choice([-1, 1])
            mean += zc * rho ** np.abs(idx - k)
        zhat = mean + e
        pv = 2 * stats.norm.sf(np.abs(zhat))
        rows.append(pd.DataFrame({"chr": c, "pos": offset + idx / per_mb, "p": pv}))
        offset += mb
    df = pd.concat(rows)
    df["lp"] = -np.log10(df["p"])
    df["col"] = np.where(df["chr"] % 2 == 1, "odd", "even")
    df["sig"] = df["p"] < GWS
    mids = df.groupby("chr")["pos"].median()
    shown = [c for c in range(1, 23) if c <= 12 or c % 2 == 0]
    p_ = (ggplot(df, aes("pos", "lp"))
          + geom_point(aes(color="col"), size=0.4, raster=True, data=df[~df["sig"]])
          + geom_point(color=PAL[1], size=0.6, raster=True, data=df[df["sig"]])
          + geom_hline(yintercept=-np.log10(GWS), color="grey", size=0.5)
          + scale_color_manual(values={"odd": "#555555", "even": "#aaaaaa"}, guide=None)
          + scale_x_continuous(breaks=[mids[c] for c in shown], labels=[str(c) for c in shown],
                               expand=(0.01, 0))
          + labs(x="chromosome", y="$-\\log_{10} p$")
          + THEME + theme(panel_grid_major_x=element_blank()))
    save(p_, "manhattan.pdf", 6.0, 2.8)
    print(f"manhattan: {len(df)} SNPs, {df['sig'].sum()} genome-wide significant")


# ---------------------------------------------------------------- LD

def fig_ld_heatmap():
    """r^2 between SNPs in two simulated populations sharing recombination hotspots."""
    rng = np.random.default_rng(3)
    P = 300
    hot = [70, 150, 230]
    founders_big = random_founders(rng, 60, P, blocks_from_hotspots(P, hot))
    # the bottlenecked population descends from a few of the same founders
    founders_small = founders_big[rng.choice(60, 5, replace=False)]
    H_big = mosaic_haplotypes(rng, 2000, founders_big, switch_profile(P, 0.05, 0.9, hot))
    H_small = mosaic_haplotypes(rng, 2000, founders_small, switch_profile(P, 0.002, 0.5, hot))
    G_big, G_small = genotypes_from_haps(H_big), genotypes_from_haps(H_small)
    keep = (maf(G_big) > 0.05) & (maf(G_small) > 0.05)
    pos = np.arange(P)[keep]
    dfs = []
    for G, name in [(G_big, "large ancestral population"), (G_small, "after a bottleneck")]:
        R2 = np.corrcoef(G[:, keep].T) ** 2
        ii, jj = np.meshgrid(np.arange(keep.sum()), np.arange(keep.sum()), indexing="ij")
        dfs.append(pd.DataFrame({"i": ii.ravel(), "j": jj.ravel(), "r2": R2.ravel(), "pop": name}))
    df = pd.concat(dfs)
    df = df[df["j"] >= df["i"]]
    df["pop"] = pd.Categorical(df["pop"], ["large ancestral population", "after a bottleneck"])
    hot_idx = [np.searchsorted(pos, h + 1) - 0.5 for h in hot]
    p_ = (ggplot(df, aes("j", "i", fill="r2"))
          + geom_tile()
          + scale_fill_gradient(low="#ffffff", high="#c0152f", limits=(0, 1), name="$r^2$")
          + facet_wrap("~pop")
          + annotate("point", x=hot_idx, y=hot_idx, shape="v", size=2.5, color=PAL[0])
          + scale_y_reverse_()
          + coord_equal()
          + labs(x="SNP", y="SNP")
          + THEME + theme(legend_title=element_text(), panel_grid=element_blank()))
    save(p_, "ld_heatmap.pdf", 6.0, 3.3)
    print(f"ld_heatmap: {keep.sum()} SNPs; mean r2 adjacent big "
          f"{np.mean(np.diag(np.corrcoef(G_big[:, keep].T)**2, 1)):.2f}, small "
          f"{np.mean(np.diag(np.corrcoef(G_small[:, keep].T)**2, 1)):.2f}")


def scale_y_reverse_():
    from plotnine import scale_y_reverse
    return scale_y_reverse()


# ---------------------------------------------------------------- architecture

def ncp_for_power(power=0.8, alpha=GWS):
    thr = stats.chi2.isf(alpha, 1)
    return brentq(lambda nc: stats.ncx2.sf(thr, 1, nc) - power, 1, 200), thr


def fig_power_maf():
    """Effect size needed for detection versus MAF, and which simulated variants are found."""
    rng = np.random.default_rng(5)
    ncp80, thr = ncp_for_power()
    f = np.logspace(-4, np.log10(0.5), 200)
    curves = []
    for N in [1e4, 1e5, 1e6]:
        curves.append(pd.DataFrame({"maf": f, "beta": np.sqrt(ncp80 / (N * 2 * f * (1 - f))),
                                    "N": f"N = {int(N):,}"}))
    curves = pd.concat(curves)
    curves["N"] = pd.Categorical(curves["N"], ["N = 10,000", "N = 100,000", "N = 1,000,000"])
    # causal variants: MAF from the folded neutral spectrum on [1e-4, 0.5]
    M, h2, N = 5000, 0.5, 1e5
    grid = np.logspace(-4, np.log10(0.5), 4000)
    dens = (1 / grid + 1 / (1 - grid)) * grid  # density on log scale
    pm = rng.choice(grid, M, p=dens / dens.sum())
    pts = []
    for S, lab in [(0.0, "no selection ($S=0$)"), (-1.0, "negative selection ($S=-1$)")]:
        v = (2 * pm * (1 - pm)) ** S
        v *= h2 / np.sum(v * 2 * pm * (1 - pm))
        beta = rng.normal(0, np.sqrt(v))
        ncp = N * 2 * pm * (1 - pm) * beta ** 2
        found = rng.random(M) < stats.ncx2.sf(thr, 1, ncp)
        pts.append(pd.DataFrame({"maf": pm[found], "beta": np.abs(beta[found]), "panel": lab}))
        print(f"power_maf {lab}: {found.sum()} discovered; "
              f"{np.sum(found & (pm < 0.01))} with MAF<1%")
    pts = pd.concat(pts)
    order = ["no selection ($S=0$)", "negative selection ($S=-1$)"]
    pts["panel"] = pd.Categorical(pts["panel"], order)
    cur = pd.concat([curves.assign(panel=o) for o in order])
    cur["panel"] = pd.Categorical(cur["panel"], order)
    cur = cur[cur["beta"] < 3]
    p_ = (ggplot()
          + geom_point(aes("maf", "beta"), data=pts, color=PAL[0], size=0.9, alpha=0.6)
          + geom_line(aes("maf", "beta", linetype="N"), data=cur, color=PAL[4])
          + facet_wrap("~panel")
          + scale_x_log10(breaks=[1e-3, 1e-2, 0.1, 0.5], labels=["0.1", "1", "10", "50"])
          + scale_y_log10()
          + labs(x="minor allele frequency (%)", y="|effect| per allele (phenotype SD)")
          + THEME + theme(legend_position="top"))
    save(p_, "power_maf.pdf", 6.0, 3.4)


# ---------------------------------------------------------------- population structure

def sim_structured_cohort(rng, P, p_anc, fst, n_unrel, n_fam):
    """Genotypes for two populations, each with n_unrel unrelated people and n_fam sib pairs."""
    a = p_anc * (1 - fst) / fst
    b = (1 - p_anc) * (1 - fst) / fst
    Gs, pop, fam = [], [], []
    for k in range(2):
        pk = rng.beta(a, b)
        n_par = n_unrel + 2 * n_fam
        par = rng.binomial(2, pk, size=(n_par, P))
        unrel = par[:n_unrel]
        mom, dad = par[n_unrel::2], par[n_unrel + 1::2]
        kids = []
        for _ in range(2):
            kids.append(rng.binomial(1, mom / 2) + rng.binomial(1, dad / 2))
        sibs = np.empty((2 * n_fam, P), dtype=int)
        sibs[0::2], sibs[1::2] = kids[0], kids[1]
        Gs.append(np.vstack([unrel, sibs]))
        pop += [k] * (n_unrel + 2 * n_fam)
        fam += [-1] * n_unrel + list(np.repeat(np.arange(n_fam) + 1000 * k, 2))
    return np.vstack(Gs), np.array(pop), np.array(fam)


def lmm_null_delta(yt, Ct, s):
    """ML estimate of delta = sigma_e^2 / sigma_g^2 in rotated coordinates, beta = covariates only."""
    N = len(yt)

    def negll(logd):
        w = 1 / (s + np.exp(logd))
        A = Ct.T @ (Ct * w[:, None])
        b = np.linalg.solve(A, Ct.T @ (w * yt))
        r = yt - Ct @ b
        sg2 = np.sum(w * r ** 2) / N
        return 0.5 * (N * np.log(sg2) - np.sum(np.log(w)) + N)

    grid = np.linspace(-5, 5, 101)
    vals = [negll(g) for g in grid]
    return np.exp(grid[int(np.argmin(vals))])


def lmm_scan(yt, Xt, Ct, s, delta):
    """GLS test of each column of Xt given rotated covariates Ct and eigenvalues s."""
    w = 1 / (s + delta)
    N = len(yt)
    z = np.empty(Xt.shape[1])
    for j in range(Xt.shape[1]):
        A = np.column_stack([Ct, Xt[:, j]])
        AtWA = A.T @ (A * w[:, None])
        b = np.linalg.solve(AtWA, A.T @ (w * yt))
        r = yt - A @ b
        sg2 = np.sum(w * r ** 2) / (N - A.shape[1])
        cov = sg2 * np.linalg.inv(AtWA)
        z[j] = b[-1] / np.sqrt(cov[-1, -1])
    return z


def fig_popstrat_qq():
    rng = np.random.default_rng(8)
    P_bg, P_test, fst = 5000, 3000, 0.01
    p_anc = rng.uniform(0.05, 0.95, P_bg + P_test)
    G, pop, fam = sim_structured_cohort(rng, P_bg + P_test, p_anc, fst, n_unrel=200, n_fam=150)
    N = len(pop)
    Z = standardize(G)
    Zbg, Ztest = Z[:, :P_bg], Z[:, P_bg:]
    causal = rng.choice(P_bg, 500, replace=False)
    h2 = 0.5
    g = Zbg[:, causal] @ rng.normal(0, np.sqrt(h2 / 500), 500)
    y = g - 0.3 * pop + rng.normal(0, np.sqrt(1 - h2), N)
    y = y - y.mean()
    print(f"popstrat: mean difference in y between populations {y[pop == 1].mean() - y[pop == 0].mean():.2f}, "
          f"in g {g[pop == 1].mean() - g[pop == 0].mean():.2f}")
    res = {}
    # (a) naive per-SNP regression
    res["linear regression"] = marginal_z(Ztest, y)
    # (b) regress out the top two genotype PCs
    U_, S_, _ = np.linalg.svd(Zbg, full_matrices=False)
    pcs = U_[:, :2]
    C = np.column_stack([np.ones(N), pcs])
    proj = lambda M: M - C @ np.linalg.lstsq(C, M, rcond=None)[0]
    yr, Xr = proj(y), proj(Ztest)
    r = (Xr.T @ yr) / np.sqrt((Xr ** 2).sum(0) * (yr ** 2).sum())
    dfree = N - C.shape[1] - 1
    res["+ 2 genotype PCs"] = r * np.sqrt(dfree / (1 - r ** 2))
    # (c) linear mixed model with realized relatedness K = Zbg Zbg^T / P_bg
    K = Zbg @ Zbg.T / P_bg
    s, U = np.linalg.eigh(K)
    s = np.clip(s, 0, None)
    yt, Xt, Ct = U.T @ y, U.T @ Ztest, U.T @ np.ones((N, 1))
    delta = lmm_null_delta(yt, Ct, s)
    res["linear mixed model"] = lmm_scan(yt, Xt, Ct, s, delta)
    print(f"popstrat: N={N}, delta={delta:.2f} (h2_hat={1/(1+delta):.2f})")
    rows = []
    order = []
    for k, z in res.items():
        lam = np.median(z ** 2) / stats.chi2.ppf(0.5, 1)
        lab = f"{k} ($\\lambda_{{GC}}={lam:.2f}$)"
        order.append(lab)
        pv = np.sort(2 * stats.norm.sf(np.abs(z)))
        ex = (np.arange(1, len(pv) + 1) - 0.5) / len(pv)
        rows.append(pd.DataFrame({"exp": -np.log10(ex), "obs": -np.log10(pv), "method": lab}))
        print(f"  {k}: lambda_GC = {lam:.3f}")
    df = pd.concat(rows)
    df["method"] = pd.Categorical(df["method"], order)
    p_ = (ggplot(df, aes("exp", "obs", color="method"))
          + geom_abline(intercept=0, slope=1, linetype="dashed", color="grey")
          + geom_point(size=0.8, raster=True)
          + scale_color_manual(values=[PAL[1], PAL[3], PAL[0]])
          + guides(color=guide_legend(ncol=1))
          + labs(x="expected $-\\log_{10} p$", y="observed $-\\log_{10} p$")
          + THEME + theme(legend_position=(0.03, 0.97), legend_justification=(0, 1),
                          legend_background=element_blank()))
    save(p_, "popstrat_qq.pdf", 4.2, 3.6)


# ---------------------------------------------------------------- fine-mapping

def log_abf(z, V, W):
    """Wakefield approximate log Bayes factor for beta ~ N(0, W) given z and V = se^2."""
    return 0.5 * np.log(V / (V + W)) + 0.5 * z ** 2 * W / (V + W)


def credible_set(pip, level=0.95):
    o = np.argsort(-pip)
    k = int(np.searchsorted(np.cumsum(pip[o]), level)) + 1
    return o[:k]


def fig_finemap():
    rng = np.random.default_rng(26)  # a locus where the lead SNP is not the causal one
    P = 220
    hot = [50, 110, 170]
    founders = random_founders(rng, 25, P, blocks_from_hotspots(P, hot))
    H = mosaic_haplotypes(rng, 10000, founders, switch_profile(P, 0.01, 0.8, hot))
    G = genotypes_from_haps(H)
    keep = maf(G) > 0.02
    G = G[:, keep]
    Xs = standardize(G)
    N, P = Xs.shape
    R = np.corrcoef(Xs.T)
    pos = np.arange(P)
    block = blocks_from_hotspots(keep.size, hot)[keep]
    # choose a causal SNP inside the second block that has several high-LD partners
    cand = [j for j in pos if block[j] == 1 and np.sum(R[j] ** 2 > 0.8) >= 4]
    c = int(rng.choice(cand))
    z_true = 7.0
    y = Xs[:, c] * z_true / np.sqrt(N) + rng.normal(0, 1, N)
    zhat = marginal_z(Xs, y)
    Ez = R[:, c] * z_true
    lead = int(np.argmax(np.abs(zhat)))
    r2lead = R[lead] ** 2
    # annotation: a few short "enhancer" segments, one containing the causal SNP
    annot = np.zeros(P, bool)
    for start in [c - 1, 15, 70, 140, 190]:
        start = min(max(start, 0), P - 3)
        annot[start:start + 3] = True
    V, W = 1.0 / N, 0.15 ** 2
    lbf = log_abf(zhat, V, W)
    pips = {}
    for name, logprior in [("uniform prior", np.zeros(P)),
                           ("annotation prior", np.log(10) * annot)]:
        a = lbf + logprior
        w = np.exp(a - a.max())
        pips[name] = w / w.sum()
    cs = {k: credible_set(v) for k, v in pips.items()}
    print(f"finemap: N={N}, P={P}, causal={c}, lead={lead}, r2(lead,causal)={R[lead, c]**2:.3f}, "
          f"n r2>0.8 with causal={np.sum(R[c]**2 > 0.8)}")
    for k in pips:
        print(f"  {k}: PIP causal={pips[k][c]:.3f}, PIP lead={pips[k][lead]:.3f}, "
              f"95% CS size={len(cs[k])}, contains causal={c in cs[k]}")
    bins = pd.cut(r2lead, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"])
    panels = ["(a) association, $-\\log_{10} p$", "(b) PIP, uniform prior", "(c) PIP, annotation prior"]
    df = pd.concat([
        pd.DataFrame({"pos": pos, "val": -np.log10(2 * stats.norm.sf(np.abs(zhat))), "r2": bins,
                      "panel": panels[0]}),
        pd.DataFrame({"pos": pos, "val": pips["uniform prior"], "r2": bins, "panel": panels[1]}),
        pd.DataFrame({"pos": pos, "val": pips["annotation prior"], "r2": bins, "panel": panels[2]}),
    ])
    df["panel"] = pd.Categorical(df["panel"], panels)
    expected = pd.DataFrame({"pos": pos, "val": -np.log10(2 * stats.norm.sf(np.abs(Ez))),
                             "panel": pd.Categorical([panels[0]] * P, panels)})
    caus = df[df["pos"] == c]
    runs = []
    j = 0
    while j < P:
        if annot[j]:
            k = j
            while k + 1 < P and annot[k + 1]:
                k += 1
            runs.append((j - 0.5, k + 0.5))
            j = k + 1
        else:
            j += 1
    rect = pd.DataFrame({"xmin": [r[0] for r in runs], "xmax": [r[1] for r in runs]})
    rect = pd.concat([rect.assign(panel=p) for p in panels])
    rect["panel"] = pd.Categorical(rect["panel"], panels)
    cols = ["#26328c", "#7fbfe5", "#5fb35f", "#f2a23a", "#d7191c"]
    p_ = (ggplot(df, aes("pos", "val"))
          + geom_rect(aes(xmin="xmin", xmax="xmax"), ymin=-np.inf, ymax=np.inf, data=rect,
                      inherit_aes=False, fill="#edae49", alpha=0.25)
          + geom_line(data=expected, color="grey", size=0.4, alpha=0.7)
          + geom_point(aes(fill="r2"), shape="o", size=1.8, stroke=0.2, color="black")
          + geom_point(data=caus, shape="o", size=4, fill="none", color="black", stroke=0.8)
          + scale_fill_manual(values=cols, name="$r^2$ with lead SNP")
          + facet_wrap("~panel", ncol=1, scales="free_y")
          + labs(x="SNP position in locus", y="")
          + THEME + theme(legend_title=element_text(size=9), legend_position="right"))
    save(p_, "finemap.pdf", 6.0, 5.4)


# ---------------------------------------------------------------- LD score regression

def fig_ldsc():
    """LD score regression on summary statistics simulated as zhat = sqrt(N) R beta + confounding + e,
    e ~ N(0, R), with R estimated from simulated haplotypes of each region."""
    rng = np.random.default_rng(13)
    n_reg, P_reg, F, n_ref = 4000, 30, 20, 2000
    N, h2 = 50000, 0.5
    bgs = [0.003, 0.01, 0.03, 0.1, 0.4]
    bsizes = [2, 3, 6, 15, 30]
    wA = rng.dirichlet(np.full(F, 2.0))
    wB = 0.9 * wA + 0.1 * rng.dirichlet(np.full(F, 2.0))
    regions = []
    for r in range(n_reg):
        blk = np.arange(P_reg) // bsizes[r % len(bsizes)]
        founders = random_founders(rng, F, P_reg, blk)
        sw = switch_profile(P_reg, bgs[(r // len(bsizes)) % len(bgs)], 0.8,
                            np.where(np.diff(blk) != 0)[0])
        GA = genotypes_from_haps(mosaic_haplotypes(rng, n_ref, founders, sw, weights=wA))
        GB = genotypes_from_haps(mosaic_haplotypes(rng, n_ref, founders, sw, weights=wB))
        G = np.vstack([GA, GB])
        keep = maf(G) > 0.05
        G, GA, GB = G[:, keep], GA[:, keep], GB[:, keep]
        R = np.corrcoef(G.T)
        # standardized allele frequency difference between the populations
        d = (GA.mean(0) - GB.mean(0)) / G.std(0)
        L = np.linalg.cholesky(R + 1e-6 * np.eye(len(R)))
        regions.append((R, d, L))
    M = sum(len(R) for R, _, _ in regions)
    ell = np.concatenate([(R ** 2).sum(1) for R, _, _ in regions])
    d_all = np.concatenate([d for _, d, _ in regions])
    sims = {}
    for name, c in [("no confounding", 0.0), ("population stratification", 0.04)]:
        z = []
        for R, d, L in regions:
            beta = rng.normal(0, np.sqrt(h2 / M), len(R))
            z.append(np.sqrt(N) * R @ beta + np.sqrt(N) * c * d + L @ rng.normal(size=len(R)))
        sims[name] = (np.concatenate(z) ** 2, ell, M)
    print(f"ldsc: M={M}, mean LD score {ell.mean():.1f}, var(d)={d_all.var():.3f}, "
          f"expected stratification term N*c^2*var(d)={N * 0.04**2 * d_all.var():.2f}")
    pts, lines, labs_ = [], [], []
    for name, (chi2, ell, M) in sims.items():
        edges = np.array([0, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6.5, 8, 10, 13, 17, 30])
        b = np.digitize(ell, edges)
        A = np.column_stack([np.ones(M), ell])
        coef = np.linalg.lstsq(A, chi2, rcond=None)[0]
        h2_hat = coef[1] * M / N
        lam = np.median(chi2) / stats.chi2.ppf(0.5, 1)
        lab = f"{name}: intercept {coef[0]:.2f}, $\\hat h^2$ = {h2_hat:.2f}"
        labs_.append(lab)
        print(f"ldsc {name}: M={M}, mean chi2={chi2.mean():.2f}, lambda_GC={lam:.2f}, "
              f"intercept={coef[0]:.3f}, slope={coef[1]:.4f}, h2_hat={h2_hat:.3f}, "
              f"LD score range {ell.min():.1f}-{ell.max():.1f}")
        ks = [k for k in np.unique(b) if np.sum(b == k) >= 500]
        pts.append(pd.DataFrame({"ell": [ell[b == k].mean() for k in ks],
                                 "chi2": [chi2[b == k].mean() for k in ks], "scen": lab}))
        xx = np.linspace(0, 16, 50)
        lines.append(pd.DataFrame({"ell": xx, "chi2": coef[0] + coef[1] * xx, "scen": lab}))
    pts, lines = pd.concat(pts), pd.concat(lines)
    for d in (pts, lines):
        d["scen"] = pd.Categorical(d["scen"], labs_)
    p_ = (ggplot(pts, aes("ell", "chi2", color="scen"))
          + geom_hline(yintercept=1, linetype="dotted", color="grey")
          + geom_line(data=lines)
          + geom_point(size=1.6)
          + scale_color_manual(values=[PAL[0], PAL[1]])
          + guides(color=guide_legend(ncol=1))
          + labs(x="LD score $\\ell_j$", y="mean $\\chi^2_j$")
          + THEME + theme(legend_position="top"))
    save(p_, "ldsc.pdf", 4.8, 3.6)


if __name__ == "__main__":
    fig_sfs()
    fig_manhattan()
    fig_ld_heatmap()
    fig_power_maf()
    fig_popstrat_qq()
    fig_finemap()
    fig_ldsc()
