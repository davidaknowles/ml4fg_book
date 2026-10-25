"""Generate the computed figures for the AlphaFold chapter.

Each function writes one PDF to ../figures/. Run from any directory:
    python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, annotate, coord_equal, element_blank, element_text, facet_wrap,
    geom_hline, geom_line, geom_point, geom_ribbon, geom_tile, ggplot, labs,
    scale_color_manual, scale_fill_gradient, scale_fill_manual, scale_linetype_manual,
    scale_x_continuous, scale_y_continuous, scale_y_reverse, theme, theme_bw,
)

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


# ---------------------------------------------------------------- coevolution

def sample_potts(rng, L, q, h, J, n_seq, n_burn=200, thin=5):
    """Gibbs sampling from p(x) ~ exp(sum_i h_i(x_i) + sum_{i<j} J_ij(x_i, x_j)).

    h: (L, q) fields; J: (L, L, q, q) couplings with J[i, j] = J[j, i].T and J[i, i] = 0.
    """
    x = rng.integers(q, size=L)
    out = []
    for sweep in range(n_burn + n_seq * thin):
        for i in rng.permutation(L):
            e = h[i] + J[i, np.arange(L), :, x].sum(0)  # (q,)
            p = np.exp(e - e.max())
            x[i] = rng.choice(q, p=p / p.sum())
        if sweep >= n_burn and (sweep - n_burn) % thin == 0:
            out.append(x.copy())
    return np.array(out)


def mutual_information(msa, q, pseudo=0.5):
    """Pairwise mutual information between MSA columns, with a pseudocount."""
    n, L = msa.shape
    oh = np.eye(q)[msa]  # (n, L, q)
    f1 = (oh.sum(0) + pseudo / q) / (n + pseudo)  # (L, q)
    f2 = (np.einsum("nia,njb->ijab", oh, oh) + pseudo / q ** 2) / (n + pseudo)
    mi = (f2 * (np.log(f2) - np.log(f1[:, None, :, None]) - np.log(f1[None, :, None, :]))).sum((2, 3))
    np.fill_diagonal(mi, 0)
    return mi


def apc(mi):
    """Average product correction (Dunn et al. 2008)."""
    L = mi.shape[0]
    off = ~np.eye(L, dtype=bool)
    row = (mi * off).sum(1) / (L - 1)
    tot = mi[off].mean()
    out = mi - np.outer(row, row) / tot
    np.fill_diagonal(out, 0)
    return out


def gibbs_steps(rng, x, h, J, n_steps):
    """n_steps single-site Gibbs updates at random positions: mutations under selection."""
    L, q = h.shape
    for _ in range(n_steps):
        i = rng.integers(L)
        e = h[i] + J[i, np.arange(L), :, x].sum(0)
        p = np.exp(e - e.max())
        x[i] = rng.choice(q, p=p / p.sum())
    return x


def tree_msa(rng, h, J, depth, n_mut):
    """Evolve sequences down a balanced binary tree; each branch applies n_mut Gibbs updates.

    Leaves (2**depth sequences) share ancestry, which induces phylogenetic correlations."""
    L, q = h.shape
    nodes = [sample_potts(rng, L, q, h, J, 1)[0]]
    for _ in range(depth):
        nodes = [gibbs_steps(rng, n.copy(), h, J, n_mut) for n in nodes for _ in range(2)]
    return np.array(nodes)


def fig_coevolution():
    """Toy MSA evolved on a tree under a Potts model with known contacts; MI and APC-corrected MI."""
    rng = np.random.default_rng(14)
    L, q = 40, 5
    contacts = [(2, 20), (20, 33), (5, 27), (8, 38), (14, 30), (30, 36), (10, 24), (17, 35),
                (22, 39), (0, 13), (6, 18), (9, 25)]
    used = {a for c in contacts for a in c}
    # fields: a few strongly conserved positions, the rest weakly biased
    h = rng.normal(0, 0.3, (L, q))
    conserved = [k for k in range(L) if k not in used][:6]
    for k in conserved:
        h[k, rng.integers(q)] += 3.5
    J = np.zeros((L, L, q, q))
    for i, j in contacts:
        compat = rng.permutation(q)  # residue a at i pairs with residue compat[a] at j
        C = np.full((q, q), -0.4)
        C[np.arange(q), compat] = 1.2
        J[i, j], J[j, i] = C, C.T
    msa = tree_msa(rng, h, J, depth=8, n_mut=6)
    mi = mutual_information(msa, q)
    mic = apc(mi)
    true = np.zeros((L, L))
    for i, j in contacts:
        true[i, j] = true[j, i] = 1

    def top_precision(m, k):
        iu = np.triu_indices(L, 1)
        order = np.argsort(-m[iu])[:k]
        return true[iu][order].mean()

    k = len(contacts)
    print(f"  coevolution: {k} contacts; top-{k} precision MI {top_precision(mi, k):.2f}, "
          f"APC {top_precision(mic, k):.2f}")
    # per-position entropy, to show conserved columns are not the contacts
    f1 = np.eye(q)[msa].mean(0)
    ent = -(np.where(f1 > 0, f1 * np.log(f1), 0)).sum(1)
    print("  conserved columns", conserved, "entropy", np.round(ent[conserved], 2))

    rows = []
    for name, m in [("true contacts", true), ("mutual information", mi / mi.max()),
                    ("MI with APC", np.clip(mic, 0, None) / mic.max())]:
        ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        rows.append(pd.DataFrame({"i": ii.ravel() + 1, "j": jj.ravel() + 1, "v": m.ravel(), "panel": name}))
    df = pd.concat(rows)
    df["panel"] = pd.Categorical(df["panel"], ["true contacts", "mutual information", "MI with APC"])
    p = (ggplot(df, aes("j", "i", fill="v")) + geom_tile()
         + scale_fill_gradient(low="white", high=PAL[0], guide=None)
         + facet_wrap("~panel") + coord_equal() + scale_y_reverse()
         + labs(x="position $j$", y="position $i$") + THEME)
    save(p, "coevolution.pdf", 6.5, 2.6)
    return msa, contacts, conserved


# ---------------------------------------------------------------- backbone geometry

BOND = {"N-CA": 1.458, "CA-C": 1.525, "C-N": 1.329}
ANGLE = {"N-CA-C": 111.2, "CA-C-N": 116.2, "C-N-CA": 121.7}


def place(a, b, c, bond, angle, torsion):
    """NeRF: position of atom d given a, b, c, |cd|, angle bcd and torsion abcd (degrees)."""
    angle, torsion = np.deg2rad(angle), np.deg2rad(torsion)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.stack([bc, np.cross(n, bc), n], 1)
    d2 = bond * np.array([-np.cos(angle), np.sin(angle) * np.cos(torsion), np.sin(angle) * np.sin(torsion)])
    return c + m @ d2


def build_backbone(phi, psi, omega=180.0):
    """Backbone N, CA, C coordinates (R x 3 x 3) from torsion angles (degrees)."""
    R = len(phi)
    N = np.array([0.0, 0.0, 0.0])
    CA = np.array([BOND["N-CA"], 0.0, 0.0])
    a = np.deg2rad(ANGLE["N-CA-C"])
    C = CA + BOND["CA-C"] * np.array([-np.cos(a), np.sin(a), 0.0])
    X = [[N, CA, C]]
    for r in range(1, R):
        N = place(X[-1][0], X[-1][1], X[-1][2], BOND["C-N"], ANGLE["CA-C-N"], psi[r - 1])
        CA = place(X[-1][1], X[-1][2], N, BOND["N-CA"], ANGLE["C-N-CA"], omega)
        C = place(X[-1][2], N, CA, BOND["CA-C"], ANGLE["N-CA-C"], phi[r])
        X.append([N, CA, C])
    return np.array(X)


def frames(X):
    """AlphaFold backbone frames by Gram-Schmidt: rotations (R, 3, 3) and translations (R, 3)."""
    N, CA, C = X[:, 0], X[:, 1], X[:, 2]
    v1, v2 = C - CA, N - CA
    e1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    u2 = v2 - (e1 * v2).sum(1, keepdims=True) * e1
    e2 = u2 / np.linalg.norm(u2, axis=1, keepdims=True)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], 2), CA  # columns of R are the local axes


def fape(Xp, Xt, clamp=10.0, eps=1e-4):
    """Frame-aligned point error in Angstrom (before division by Z), over all backbone atoms."""
    Rp, tp = frames(Xp)
    Rt, tt = frames(Xt)
    P, T = Xp.reshape(-1, 3), Xt.reshape(-1, 3)
    # local coordinates of every atom j in every frame i: R_i^T (x_j - t_i)
    lp = np.einsum("iba,ijb->ija", Rp, P[None] - tp[:, None])
    lt = np.einsum("iba,ijb->ija", Rt, T[None] - tt[:, None])
    d = np.sqrt(((lp - lt) ** 2).sum(-1) + eps)
    return np.minimum(d, clamp).mean()


def drmsd(Xp, Xt):
    """RMS difference of pairwise C-alpha distances."""
    def dm(X):
        ca = X[:, 1]
        return np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    iu = np.triu_indices(len(Xp), 1)
    return np.sqrt(((dm(Xp) - dm(Xt))[iu] ** 2).mean())


def kabsch_rmsd(Xp, Xt):
    """C-alpha RMSD after optimal superposition by a proper rotation and translation."""
    A, B = Xp[:, 1] - Xp[:, 1].mean(0), Xt[:, 1] - Xt[:, 1].mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    Rm = U @ D @ Vt
    return np.sqrt((((A @ Rm) - B) ** 2).sum(1).mean())


def random_rotation(rng):
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rr)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def helix(R=30):
    return build_backbone(np.full(R, -57.8), np.full(R, -47.0))


def fig_fape():
    """FAPE, dRMSD and superposed RMSD for perturbed and mirrored copies of an ideal alpha helix."""
    rng = np.random.default_rng(3)
    Xt = helix(30)
    ca = Xt[:, 1]
    print(f"  helix: CA(i)-CA(i+4) {np.linalg.norm(ca[4:] - ca[:-4], axis=1).mean():.2f} A, "
          f"CA(i)-CA(i+1) {np.linalg.norm(ca[1:] - ca[:-1], axis=1).mean():.2f} A")
    # global rotation and translation: every metric is zero
    Q = random_rotation(rng)
    Xg = Xt @ Q.T + rng.normal(0, 20, 3)
    assert fape(Xg, Xt) < 0.02 and drmsd(Xg, Xt) < 1e-8 and kabsch_rmsd(Xg, Xt) < 1e-6
    # mirror image
    Xm = Xt * np.array([1, 1, -1])
    mir = {"FAPE": fape(Xm, Xt), "dRMSD": drmsd(Xm, Xt), "RMSD": kabsch_rmsd(Xm, Xt)}
    print("  mirror image:", {k: round(v, 2) for k, v in mir.items()})
    rows = []
    R = len(Xt)
    for s in np.linspace(0, 20, 11):
        for rep in range(30):
            # perturb every backbone torsion angle and rebuild with ideal bond geometry
            Xp = build_backbone(-57.8 + rng.normal(0, s, R), -47.0 + rng.normal(0, s, R))
            Xp = Xp @ random_rotation(rng).T  # predictions live in an arbitrary global frame
            rows += [(s, "FAPE", fape(Xp, Xt)), (s, "dRMSD", drmsd(Xp, Xt)),
                     (s, "RMSD", kabsch_rmsd(Xp, Xt))]
    df = pd.DataFrame(rows, columns=["sigma", "metric", "err"])
    sm = df.groupby(["sigma", "metric"], as_index=False)["err"].agg(["mean", "std"]).reset_index()
    sm["lo"], sm["hi"] = sm["mean"] - sm["std"], sm["mean"] + sm["std"]
    levels = ["FAPE", "RMSD", "dRMSD"]
    sm["metric"] = pd.Categorical(sm["metric"], levels)
    cols = [PAL[1], PAL[0], PAL[2]]
    p = (ggplot(sm, aes("sigma", "mean", color="metric", fill="metric"))
         + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.15, color="none", show_legend=False)
         + geom_line(size=0.8))
    for k, c in zip(levels, cols):
        p = p + geom_hline(yintercept=mir[k], color=c, linetype="dashed")
    p = (p + annotate("text", x=0.3, y=mir["FAPE"] + 0.3, label="mirror image (dashed)", ha="left", size=8)
         + scale_color_manual(values=cols) + scale_fill_manual(values=cols, guide=None)
         + labs(x="torsion-angle noise s.d. (degrees)", y="error (\u00c5)")
         + THEME + theme(legend_position="right"))
    save(p, "fape_vs_rmsd.pdf", 4.8, 3.0)
    return mir


def fig_triangle_violations():
    """Fraction of residue triples violating the triangle inequality when pairwise distances are
    predicted independently with noise, for an ideal helix."""
    rng = np.random.default_rng(7)
    Xt = helix(30)
    ca = Xt[:, 1]
    D = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    L = len(ca)
    i, j, k = np.array([(a, b, c) for a in range(L) for b in range(a + 1, L) for c in range(b + 1, L)]).T
    rows = []
    for s in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        for rep in range(20):
            E = rng.normal(0, s, D.shape)
            Dn = np.abs(D + np.triu(E, 1) + np.triu(E, 1).T)
            a, b, c = Dn[i, j], Dn[i, k], Dn[j, k]
            viol = (a > b + c) | (b > a + c) | (c > a + b)
            rows.append((s, viol.mean()))
    df = pd.DataFrame(rows, columns=["sigma", "frac"]).groupby("sigma", as_index=False)["frac"].mean()
    print("  triangle violations:", df.round(3).to_dict("records"))
    p = (ggplot(df, aes("sigma", "frac")) + geom_line(size=0.8, color=PAL[0]) + geom_point(size=1.5, color=PAL[0])
         + labs(x="noise s.d. per distance (Å)", y="fraction of triples violating")
         + THEME)
    save(p, "triangle_violations.pdf", 3.2, 2.7)


if __name__ == "__main__":
    for fn in [fig_coevolution, fig_fape, fig_triangle_violations]:
        fn()
        print("wrote", fn.__name__)
