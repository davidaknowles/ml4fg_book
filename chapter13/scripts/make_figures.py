"""Generate the computed figures for the rare variants chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, annotate, element_blank, element_text, facet_grid, facet_wrap, geom_col,
    geom_histogram, geom_line, geom_point, geom_text, geom_vline, ggplot, labs,
    position_dodge, scale_color_manual, scale_fill_manual, scale_linetype_manual,
    scale_x_log10, scale_y_continuous, theme, theme_bw,
)
from scipy.special import betaln, gammaln, expit
from scipy.stats import beta as beta_dist, norm

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


# ---------------------------------------------------------------- single-variant power

def fig_power_single_variant():
    """Power of a single-variant additive test at genome-wide significance, as a function of MAF.

    Quantitative trait y = beta g + eps with eps ~ N(0, sigma^2) and g ~ Binomial(2, f).
    The z statistic has non-centrality sqrt(n 2 f (1 - f)) beta / sigma.
    """
    alpha = 5e-8
    zc = norm.isf(alpha / 2)
    f = np.logspace(-5, np.log10(0.5), 400)
    rows = []
    for beta in [0.1, 0.5]:
        for n in [1e3, 1e4, 1e5, 1e6]:
            ncp = np.sqrt(n * 2 * f * (1 - f)) * beta
            power = norm.sf(zc - ncp) + norm.cdf(-zc - ncp)
            rows.append(pd.DataFrame({"f": f, "power": power,
                                      "n": f"$n=10^{{{int(np.log10(n))}}}$",
                                      "beta": f"$\\beta/\\sigma={beta:g}$"}))
    df = pd.concat(rows)
    df["n"] = pd.Categorical(df["n"], [f"$n=10^{{{k}}}$" for k in [3, 4, 5, 6]])
    df["beta"] = pd.Categorical(df["beta"], ["$\\beta/\\sigma=0.1$", "$\\beta/\\sigma=0.5$"])
    p = (ggplot(df, aes("f", "power", color="n")) + geom_line(size=0.8)
         + facet_wrap("~beta")
         + scale_x_log10(breaks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                         labels=["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
         + scale_color_manual(values=PAL[:4])
         + labs(x="minor allele frequency $f$", y="power at $\\alpha=5\\times10^{-8}$")
         + THEME + theme(legend_position="right"))
    save(p, "power_single_variant.pdf", 6.0, 2.8)


# ---------------------------------------------------------------- weights

def fig_maf_weights():
    """Variant weights as a function of MAF: Beta(1,25) density (SKAT default), Beta(1,1) and Madsen-Browning."""
    f = np.logspace(-4, np.log10(0.5), 400)
    rows = [
        pd.DataFrame({"f": f, "w": beta_dist.pdf(f, 1, 25), "scheme": "Beta(1, 25)"}),
        pd.DataFrame({"f": f, "w": np.ones_like(f), "scheme": "Beta(1, 1) (unweighted)"}),
        pd.DataFrame({"f": f, "w": 1 / np.sqrt(f * (1 - f)), "scheme": "$1/\\sqrt{f(1-f)}$"}),
    ]
    df = pd.concat(rows)
    df["w"] = df["w"] / df.groupby("scheme")["w"].transform("max")
    df["scheme"] = pd.Categorical(df["scheme"], ["Beta(1, 25)", "$1/\\sqrt{f(1-f)}$",
                                                 "Beta(1, 1) (unweighted)"])
    p = (ggplot(df, aes("f", "w", color="scheme", linetype="scheme")) + geom_line(size=0.8)
         + scale_x_log10(breaks=[1e-4, 1e-3, 1e-2, 1e-1],
                         labels=["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
         + scale_color_manual(values=[PAL[0], PAL[1], PAL[4]])
         + scale_linetype_manual(values=["solid", "dashed", "dotted"])
         + labs(x="minor allele frequency $f_j$", y="weight $w_j$ (scaled to max 1)")
         + THEME + theme(legend_position="right"))
    save(p, "maf_weights.pdf", 5.5, 2.6)


# ---------------------------------------------------------------- set-based tests

RHOS = np.array([0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.5, 1.0])


def simulate_region(rng, n=2000, m=40):
    f = 10 ** rng.uniform(np.log10(5e-4), np.log10(0.02), m)
    G = rng.binomial(2, f, size=(n, m)).astype(float)
    keep = G.sum(0) > 0
    return G[:, keep], f[keep]


def q_stats(S, w, rhos=RHOS):
    """Q_rho = (1 - rho) sum_j w_j^2 S_j^2 + rho (sum_j w_j S_j)^2 for each row of S (B x m)."""
    ws = S * w
    q_skat = (ws ** 2).sum(1)
    q_burden = ws.sum(1) ** 2
    return (1 - rhos)[None, :] * q_skat[:, None] + rhos[None, :] * q_burden[:, None]


def emp_p(null_sorted, x):
    """Upper-tail empirical p-values of x against each column's sorted null draws."""
    B = null_sorted.shape[0]
    out = np.empty_like(x)
    for k in range(x.shape[1]):
        out[:, k] = (B - np.searchsorted(null_sorted[:, k], x[:, k], side="left") + 1) / (B + 1)
    return out


class SetTests:
    """Burden, SKAT and SKAT-O (and multi-weight min-p) with a simulated null for a fixed design.

    Under H0, with y centered and sigma known, the score vector S = G_c^T y / sigma is N(0, G_c^T G_c).
    """

    def __init__(self, G, weights, rng, B=200_000):
        self.Gc = G - G.mean(0)
        self.weights = weights  # list of weight vectors (one per annotation)
        L = np.linalg.cholesky(self.Gc.T @ self.Gc + 1e-9 * np.eye(G.shape[1]))
        S0 = rng.standard_normal((B, G.shape[1])) @ L.T
        self.null = [np.sort(q_stats(S0, w), 0) for w in weights]
        # null distribution of the min-p statistics, computed with p-values from the same null
        p0 = [emp_p(nl, q_stats(S0, w)) for nl, w in zip(self.null, weights)]
        self.null_min = {
            "burden": np.sort(np.min([p[:, -1] for p in p0], 0)),
            "SKAT": np.sort(np.min([p[:, 0] for p in p0], 0)),
            "SKAT-O": np.sort(np.min([p.min(1) for p in p0], 0)),
        }

    def pvalues(self, y):
        yc = (y - y.mean()) / y.std()
        S = (self.Gc.T @ yc)[None, :]
        ps = [emp_p(nl, q_stats(S, w)) for nl, w in zip(self.null, self.weights)]
        out = {}
        for name, stat in [("burden", lambda p: p[:, -1]), ("SKAT", lambda p: p[:, 0]),
                           ("SKAT-O", lambda p: p.min(1))]:
            t = np.min([stat(p) for p in ps], 0)
            nm = self.null_min[name]
            out[name] = (np.searchsorted(nm, t, side="right") + 1) / (len(nm) + 1)
        return out


def fig_skat_power():
    """Power of burden, SKAT and SKAT-O as the fraction of causal variants and their signs vary."""
    rng = np.random.default_rng(13)
    G, f = simulate_region(rng)
    w = beta_dist.pdf(f, 1, 25)
    tests = SetTests(G, [w], rng)
    alpha, R = 1e-3, 1000
    rows = []
    for signs in ["all risk-increasing", "50% protective"]:
        for frac in [0.1, 0.3, 0.6, 1.0]:
            a = {0.1: 0.4, 0.3: 0.15, 0.6: 0.085, 1.0: 0.06}[frac]
            hits = {k: 0 for k in ["burden", "SKAT", "SKAT-O"]}
            for _ in range(R):
                causal = rng.random(len(f)) < frac
                b = a * np.abs(np.log10(f)) * causal
                if signs != "all risk-increasing":
                    b *= rng.choice([-1, 1], len(f))
                y = G @ b + rng.standard_normal(G.shape[0])
                for k, pv in tests.pvalues(y).items():
                    hits[k] += pv[0] < alpha
            for k, v in hits.items():
                rows.append({"signs": signs, "frac": f"{int(frac * 100)}%", "test": k,
                             "power": v / R})
    df = pd.DataFrame(rows)
    df["frac"] = pd.Categorical(df["frac"], ["10%", "30%", "60%", "100%"])
    df["test"] = pd.Categorical(df["test"], ["burden", "SKAT", "SKAT-O"])
    df["signs"] = pd.Categorical(df["signs"], ["all risk-increasing", "50% protective"])
    df.to_csv(OUT.parent / "scripts" / "skat_power.csv", index=False)
    p = (ggplot(df, aes("frac", "power", fill="test"))
         + geom_col(position=position_dodge(width=0.8), width=0.75)
         + facet_wrap("~signs") + scale_fill_manual(values=[PAL[3], PAL[0], PAL[2]])
         + scale_y_continuous(limits=(0, 1))
         + labs(x="fraction of variants that are causal", y=f"power at $\\alpha=10^{{-3}}$")
         + THEME + theme(legend_position="top"))
    save(p, "skat_power.pdf", 6.0, 3.0)


def fig_fst_power(a=0.28, R=1000, write=True):
    """Power of tests using MAF weights only (O), functional weights (W) or both (U, min-p)."""
    rng = np.random.default_rng(17)
    G, f = simulate_region(rng)
    m = len(f)
    w = beta_dist.pdf(f, 1, 25)
    alpha = 1e-3
    rows = []
    for P in [0.0, 0.5, 1.0]:
        hits = {(k, t): 0 for k in ["O", "W", "U"] for t in ["SKAT", "burden"]}
        # the functional score is resampled for each gene-level design, so the null is rebuilt per draw set
        n_designs = 50
        for d in range(n_designs):
            causal = np.zeros(m, bool)
            causal[rng.choice(m, max(2, int(0.2 * m)), replace=False)] = True
            informative = causal & (rng.random(m) < P)
            Z = np.where(informative, rng.noncentral_chisquare(1, 4, m), rng.chisquare(1, m))
            Z = Z / Z.max()
            tO = SetTests(G, [w], rng, B=50_000)
            tW = SetTests(G, [w * Z], rng, B=50_000)
            tU = SetTests(G, [w, w * Z], rng, B=50_000)
            for _ in range(R // n_designs):
                b = a * np.abs(np.log10(f)) * causal
                y = G @ b + rng.standard_normal(G.shape[0])
                for name, t in [("O", tO), ("W", tW), ("U", tU)]:
                    pv = t.pvalues(y)
                    for test in ["SKAT", "burden"]:
                        hits[(name, test)] += pv[test][0] < alpha
        for (name, test), v in hits.items():
            rows.append({"P": f"$P={P:g}$", "weights": name, "test": test, "power": v / R})
    df = pd.DataFrame(rows)
    df["weights"] = pd.Categorical(df["weights"], ["O", "W", "U"])
    df["test"] = pd.Categorical(df["test"], ["SKAT", "burden"])
    df["P"] = pd.Categorical(df["P"], ["$P=0$", "$P=0.5$", "$P=1$"])
    if not write:
        return df
    df.to_csv(OUT.parent / "scripts" / "fst_power.csv", index=False)
    p = (ggplot(df, aes("test", "power", fill="weights"))
         + geom_col(position=position_dodge(width=0.8), width=0.75)
         + facet_wrap("~P") + scale_fill_manual(values=[PAL[0], PAL[1], PAL[2]])
         + scale_y_continuous(limits=(0, 1))
         + labs(x="", y=f"power at $\\alpha=10^{{-3}}$")
         + THEME + theme(legend_position="top"))
    save(p, "fst_power.pdf", 6.0, 2.9)


# ---------------------------------------------------------------- outliers

def fig_outliers():
    """Expression and splicing outliers: one patient against a reference population."""
    rng = np.random.default_rng(5)
    n = 500
    z = rng.standard_normal(n)
    psi = rng.beta(2, 60, n)
    pop = pd.concat([
        pd.DataFrame({"x": z, "panel": "(a) expression, $z$-score"}),
        pd.DataFrame({"x": psi, "panel": "(b) splicing, exon-skipping junction usage"}),
    ])
    pat = pd.DataFrame({"x": [-5.2, 0.42],
                        "panel": ["(a) expression, $z$-score",
                                  "(b) splicing, exon-skipping junction usage"]})
    p = (ggplot(pop, aes("x")) + geom_histogram(bins=30, fill=PAL[0], alpha=0.8)
         + geom_vline(pat, aes(xintercept="x"), color=PAL[1], size=0.9)
         + geom_text(pat.iloc[[0]].assign(y=55, label="  patient"), aes("x", "y", label="label"),
                     color=PAL[1], ha="left", size=8)
         + geom_text(pat.iloc[[1]].assign(y=55, label="patient  "), aes("x", "y", label="label"),
                     color=PAL[1], ha="right", size=8)
         + facet_wrap("~panel", scales="free_x")
         + labs(x="", y="individuals") + THEME)
    save(p, "outliers.pdf", 6.0, 2.6)


def betabinom_logpmf(y, n, a, b):
    return (gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
            + betaln(y + a, n - y + b) - betaln(a, b))


def aneva_dot_pvalue(y, n, vg, conc=200.0, grid=401):
    """Two-sided tail probability of haplotype-1 count y out of n, marginalizing the regulatory effect.

    s ~ N(0, vg) on the log2 allelic fold change scale; p(s) = 2^s / (1 + 2^s);
    y | s ~ BetaBinomial(n, conc p, conc (1 - p)).
    """
    s = np.linspace(-6 * np.sqrt(vg), 6 * np.sqrt(vg), grid)
    ws = norm.pdf(s, 0, np.sqrt(vg))
    ws /= ws.sum()
    p = expit(s * np.log(2))
    ks = np.arange(n + 1)
    pmf = (ws[:, None] * np.exp(betabinom_logpmf(ks[None, :], n, conc * p[:, None],
                                                  conc * (1 - p[:, None])))).sum(0)
    lower, upper = pmf[: y + 1].sum(), pmf[y:].sum()
    return min(1.0, 2 * min(lower, upper))


def fig_aneva_dot():
    """Population distribution of allelic counts under a gene's genetic regulatory variance, with one patient."""
    rng = np.random.default_rng(11)
    npop = 4000
    rows, pts = [], []
    patient = (640, 157)  # haplotype 1, haplotype 2 counts
    for vg, lab in [(0.02, "gene A, low $V^G$"), (1.2, "gene B, high $V^G$")]:
        N = np.round(10 ** rng.uniform(1, np.log10(1600), npop)).astype(int)
        s = rng.normal(0, np.sqrt(vg), npop)
        pr = rng.beta(200 * expit(s * np.log(2)), 200 * (1 - expit(s * np.log(2))))
        h1 = rng.binomial(N, pr)
        rows.append(pd.DataFrame({"h1": h1, "h2": N - h1, "gene": lab}))
        pv = aneva_dot_pvalue(patient[0], sum(patient), vg)
        pts.append({"h1": patient[0], "h2": patient[1], "gene": lab,
                    "label": f"$p={pv:.1g}$" if pv >= 1e-3 else f"$p={pv:.0e}$"})
    df = pd.concat(rows)
    pt = pd.DataFrame(pts)
    df["gene"] = pd.Categorical(df["gene"], ["gene A, low $V^G$", "gene B, high $V^G$"])
    pt["gene"] = pd.Categorical(pt["gene"], ["gene A, low $V^G$", "gene B, high $V^G$"])
    sq = np.sqrt
    df = df.assign(x=sq(df.h1), y=sq(df.h2))
    pt = pt.assign(x=sq(pt.h1), y=sq(pt.h2))
    brk = [0, 100, 400, 900, 1600]
    p = (ggplot(df, aes("x", "y")) + geom_point(size=0.6, alpha=0.35, color=PAL[0], stroke=0)
         + geom_point(pt, aes("x", "y"), color=PAL[1], size=2.5, shape="D")
         + geom_text(pt, aes("x", "y", label="label"), color=PAL[1], nudge_y=3.5, size=9)
         + annotate("segment", x=0, y=0, xend=40, yend=40, color="black", size=0.4)
         + facet_wrap("~gene")
         + scale_x_continuous_sqrt(brk) + scale_y_continuous(breaks=[sq(b) for b in brk],
                                                             labels=[str(b) for b in brk])
         + labs(x="haplotype 1 count", y="haplotype 2 count") + THEME)
    save(p, "aneva_dot.pdf", 6.0, 3.0)
    return pt


def scale_x_continuous_sqrt(brk):
    from plotnine import scale_x_continuous
    return scale_x_continuous(breaks=[np.sqrt(b) for b in brk], labels=[str(b) for b in brk])


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["power_single_variant", "maf_weights", "outliers", "aneva_dot",
                             "skat_power", "fst_power"]
    for name in which:
        out = globals()[f"fig_{name}"]()
        if out is not None:
            print(out)
        print("done", name)
