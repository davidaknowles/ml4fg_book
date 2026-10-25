"""Hidden Markov model with independent Bernoulli emissions (the ChromHMM emission model).

Notation follows the chapter: S positions, K states, M binary marks.
    pi : (K,)   initial state distribution
    T  : (K, K) transition matrix, T[i, j] = P(X_t = j | X_{t-1} = i)
    E  : (K, M) emission matrix, E[i, m] = P(y_tm = 1 | X_t = i)
    Y  : (S, M) binary observations
"""
import numpy as np


def simulate(pi, T, E, S, rng):
    """Draw a state path x (S,) and observations Y (S, M) from the HMM."""
    K, M = E.shape
    x = np.empty(S, dtype=int)
    x[0] = rng.choice(K, p=pi)
    for t in range(1, S):
        x[t] = rng.choice(K, p=T[x[t - 1]])
    Y = (rng.random((S, M)) < E[x]).astype(int)
    return x, Y


def log_emission(Y, E):
    """log phi_t(i) = log P(y_t | X_t = i), shape (S, K)."""
    E = np.clip(E, 1e-10, 1 - 1e-10)
    return Y @ np.log(E).T + (1 - Y) @ np.log1p(-E).T


def forward_backward(Y, pi, T, E):
    """Posterior marginals by the scaled forward-backward algorithm.

    Returns gamma (S, K) with gamma[t, i] = P(X_t = i | Y), the expected
    transition counts xi_sum (K, K) = sum_t P(X_{t-1} = i, X_t = j | Y),
    and the log likelihood log P(Y).
    """
    logphi = log_emission(Y, E)
    shift = logphi.max(1, keepdims=True)
    phi = np.exp(logphi - shift)          # phi_t up to a per-position constant
    S, K = phi.shape
    f = np.empty((S, K))
    c = np.empty(S)
    r = pi
    for t in range(S):
        q = r * phi[t]                    # q_t = r_t * phi_t (elementwise)
        c[t] = q.sum()
        f[t] = q / c[t]                   # f_t = P(X_t | y_1:t)
        r = f[t] @ T                      # r_{t+1} = T^T f_t = P(X_{t+1} | y_1:t)
    b = np.empty((S, K))
    b[-1] = 1.0
    for t in range(S - 1, 0, -1):
        b[t - 1] = T @ (phi[t] * b[t]) / c[t]
    gamma = f * b
    gamma /= gamma.sum(1, keepdims=True)
    xi_sum = np.zeros((K, K))
    for t in range(1, S):
        xi_sum += np.outer(f[t - 1], phi[t] * b[t]) * T / c[t]
    loglik = np.log(c).sum() + shift.sum()
    return gamma, xi_sum, loglik


def viterbi(Y, pi, T, E):
    """Most probable state path argmax_x P(x | Y)."""
    logphi = log_emission(Y, E)
    logT = np.log(np.clip(T, 1e-300, None))
    S, K = logphi.shape
    delta = np.log(np.clip(pi, 1e-300, None)) + logphi[0]
    back = np.zeros((S, K), dtype=int)
    for t in range(1, S):
        cand = delta[:, None] + logT       # cand[i, j]: best path ending i -> j
        back[t] = cand.argmax(0)
        delta = cand.max(0) + logphi[t]
    x = np.empty(S, dtype=int)
    x[-1] = delta.argmax()
    for t in range(S - 1, 0, -1):
        x[t - 1] = back[t, x[t]]
    return x


def mixture_posterior(Y, w, E):
    """Posterior P(X_t | y_t) for a mixture model (no transitions), weights w (K,)."""
    lp = log_emission(Y, E) + np.log(w)
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def stationary(T):
    """Stationary distribution of the Markov chain with transition matrix T."""
    vals, vecs = np.linalg.eig(T.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def baum_welch(Y, K, rng, n_iter=100, pseudo=1e-3, init=None):
    """Fit pi, T, E by EM (Baum-Welch). Returns parameters and log-likelihood trace."""
    S, M = Y.shape
    if init is None:
        pi = np.full(K, 1.0 / K)
        T = 0.8 * np.eye(K) + 0.2 * rng.dirichlet(np.ones(K), K)
        E = rng.uniform(0.05, 0.95, (K, M))
    else:
        pi, T, E = (a.copy() for a in init)
    trace = []
    for _ in range(n_iter):
        gamma, xi_sum, ll = forward_backward(Y, pi, T, E)   # E step
        trace.append(ll)
        pi = gamma[0]                                       # M step
        T = xi_sum + pseudo
        T /= T.sum(1, keepdims=True)
        E = (gamma.T @ Y + pseudo) / (gamma.sum(0)[:, None] + 2 * pseudo)
    return pi, T, E, np.array(trace)
