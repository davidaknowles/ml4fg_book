"""Attribution methods for a small sequence CNN: ISM, gradients, deconvnet, guided backprop,
DeepLIFT (Rescale rule), integrated gradients, and exact Shapley values for small functions.

The CNN uses only linear layers (convolution, average pooling, dense) and ReLUs, so every
backpropagation-based method differs from the plain gradient only in how it passes a gradient
back through a ReLU. Each method therefore swaps in a different ReLU backward rule.
"""
from itertools import combinations
from math import factorial

import numpy as np
import torch
import torch.nn as nn


class _Rule(torch.autograd.Function):
    """ReLU whose backward pass is chosen by `mode`.

    mode: "grad" (true gradient), "deconv" (pass positive gradients, ignore the forward sign),
    "guided" (both conditions), "deeplift" (Rescale rule: multiplier = delta out / delta in,
    which needs the reference pre-activation `zref`).
    """

    @staticmethod
    def forward(ctx, z, zref, mode):
        ctx.save_for_backward(z, zref)
        ctx.mode = mode
        return z.clamp(min=0)

    @staticmethod
    def backward(ctx, g):
        z, zref = ctx.saved_tensors
        on = (z > 0).to(g.dtype)
        if ctx.mode == "grad":
            return g * on, None, None
        if ctx.mode == "deconv":
            return g.clamp(min=0), None, None
        if ctx.mode == "guided":
            return g.clamp(min=0) * on, None, None
        if ctx.mode == "deeplift":
            dz = z - zref
            dout = z.clamp(min=0) - zref.clamp(min=0)
            small = dz.abs() < 1e-6
            m = torch.where(small, on, dout / torch.where(small, torch.ones_like(dz), dz))
            return g * m, None, None
        raise ValueError(ctx.mode)


class SeqCNN(nn.Module):
    """conv(4 -> F, width K) -> ReLU -> global average pool (times `scale`) -> dense -> ReLU -> dense (logit)."""

    def __init__(self, n_filters=16, width=11, hidden=16, scale=20.0):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv1d(4, n_filters, width, padding=width // 2)
        self.fc1 = nn.Linear(n_filters, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x, mode=None, ref=None):
        """x: (N, 4, L) one-hot. With mode set, ReLUs use that backward rule; DeepLIFT needs ref (1, 4, L)."""
        if mode is None:
            h = torch.relu(self.conv(x)).mean(2) * self.scale
            return self.fc2(torch.relu(self.fc1(h))).squeeze(1)
        z1 = self.conv(x)
        z1r = self.conv(ref).expand_as(z1) if mode == "deeplift" else z1
        h = _Rule.apply(z1, z1r.detach(), mode).mean(2) * self.scale
        z2 = self.fc1(h)
        if mode == "deeplift":
            z2r = self.fc1(torch.relu(self.conv(ref)).mean(2) * self.scale).expand_as(z2)
        else:
            z2r = z2
        return self.fc2(_Rule.apply(z2, z2r.detach(), mode)).squeeze(1)


def backprop_scores(model, x, mode, ref=None):
    """Gradient-like scores d f / d x under the given ReLU rule, shape (N, 4, L)."""
    x = x.clone().requires_grad_(True)
    out = model(x, mode=mode, ref=ref)
    out.sum().backward()
    return x.grad.detach()


def deeplift(model, x, ref):
    """DeepLIFT Rescale contributions (multiplier times difference from reference), shape (N, 4, L)."""
    m = backprop_scores(model, x, "deeplift", ref)
    return m * (x - ref)


def integrated_gradients(model, x, ref, steps=50):
    """Integrated gradients with a midpoint Riemann sum along the straight path from ref to x."""
    total = torch.zeros_like(x)
    for k in range(steps):
        a = (k + 0.5) / steps
        total += backprop_scores(model, ref + a * (x - ref), "grad")
    return total / steps * (x - ref)


@torch.no_grad()
def ism(model, x):
    """In silico mutagenesis. Returns (N, 4, L) with entry [b, l] = f(x with base b at l) - f(x)."""
    n, _, L = x.shape
    base = model(x)
    out = torch.zeros_like(x)
    for l in range(L):
        for b in range(4):
            xm = x.clone()
            xm[:, :, l] = 0
            xm[:, b, l] = 1
            out[:, b, l] = model(xm) - base
    return out


def shapley(f, P):
    """Exact Shapley values of a set function f(S) (S a frozenset of feature indices 0..P-1)."""
    phi = np.zeros(P)
    for i in range(P):
        others = [j for j in range(P) if j != i]
        for k in range(P):
            w = factorial(k) * factorial(P - k - 1) / factorial(P)
            for S in combinations(others, k):
                S = frozenset(S)
                phi[i] += w * (f(S | {i}) - f(S))
    return phi
