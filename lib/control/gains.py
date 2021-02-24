# coding: utf-8

import numpy as np
import torch

j = torch.view_as_complex(torch.Tensor([0, 1]).float())

def h_inf(A, B, C, D=None, epsilon=1e-4):
    assert isinstance(A, torch.Tensor), "The matrix A must be torch.Tensor, but {} is given".format(type(A))
    assert isinstance(B, torch.Tensor), "The matrix B must be torch.Tensor, but {} is given".format(type(B))
    assert isinstance(C, torch.Tensor), "The matrix C must be torch.Tensor, but {} is given".format(type(C))
    if D is None:
        D = torch.zeros_like(B)

    n, m = B.shape
    Ac = to_complex(A)
    Bc = to_complex(B)
    Cc = to_complex(C)
    Dc = to_complex(D)

    print("A:\n", Ac)
    print("B:\n", Bc)
    print("C:\n", Cc)
    print("D:\n", Dc)
    print("unit j:", j)

    def G(s):
        return Cc.mm(torch.inverse(
            s * torch.eye(n) - Ac
        )).mm(Bc) + Dc

    def H(g):
        g2 = g ** 2
        R = D.T.mm(D) - g2 * torch.eye(m)
        S = D.mm(D.T) - g2 * torch.eye(n)
        Rinv = torch.inverse(R)
        Sinv = torch.inverse(S)
        BRDC = B.mm(Rinv).mm(D.T).mm(C)
        ABRDC = A - BRDC
        ACDRB = - A.T + BRDC.T
        gBRB = -g * B.mm(Rinv).mm(B.T)
        gCSC = g * C.T.mm(Sinv).mm(C)
        return torch.vstack([
            torch.hstack([ABRDC, gBRB]),
            torch.hstack([gCSC, ACDRB])
        ])

    print("G(0)", G(0 * j))
    print("-"*60)
    print(H(1.))
    print("="*60)
    # (a) Compute a starting value for gamma_lb using (4.2)
    gamma_lb = compute_initial_gamma(G, Ac, Dc)
    print("initial gamma_lb:", gamma_lb)

    # (b) repeat until 'break'
    while True:
        # (b1)
        print("- b1 --------------------")
        gamma = gamma_lb * (1 + epsilon * 2)
        print("gamma:", gamma)
        # (b2) compute the eigenvalues of H(gamma)
        print("- b2 --------------------")
        eigs = torch.eig(H(gamma)).eigenvalues
        omegas = torch.sort(eigs[:, 1]).values
        omegas = omegas[omegas > 1e-6]
        print("all omegas:", omegas)

        # (b3) if no imaginary eigenvalues
        print("- b3 --------------------")
        domegas = omegas[1:] - omegas[:-1]
        print("omega gaps:", domegas)
        if len(omegas) == 0 or all([domega < epsilon for domega in domegas]):
            gamma_ub = gamma
            break
        ms = 0.5 * (omegas[1:] + omegas[:-1]) * j
        print("ms:", ms)

        sigs = torch.hstack([torch.max(torch.svd(G(m)).S) for m in ms])
        gamma_lb = torch.max(sigs)
        print("gamma_lb:", gamma_lb)

    gain = 0.5 * (gamma_lb + gamma_ub)
    return gain


def compute_initial_gamma(G, A, D):
    # sigma_max of G(0)
    sG0 = torch.max(torch.svd(G(0)).S)
    print("G(0):", sG0)
    # sigma_max of G(j*omega)
    # compute omega_p
    A = A if not A.is_complex() else A.real
    L = torch.eig(A)
    L = torch.view_as_complex(L.eigenvalues)
    omega_p = torch.hstack([torch.abs(l.imag / l.real / np.abs(l)) for l in L])
    omega_p = torch.max(omega_p)
    sGp = torch.max(torch.svd(G(omega_p * j)).S)
    print("G(omega_p):", sGp)
    # sigma_max of D
    sD = torch.max(torch.svd(D).S)
    print("D:", sD)

    # select maximum gamma
    gammas = torch.hstack([sG0, sGp, sD])
    return torch.max(gammas)


def to_complex(x):
    x_ = torch.unsqueeze(x, dim=-1)
    return torch.view_as_complex(torch.cat(
        (x_, torch.zeros_like(x_)), dim=-1
    ))


# def


def h_inf_garbage(A, B, C, epsilon=1e-4):
    assert isinstance(A, torch.Tensor), "The matrix A must be torch.Tensor, but {} is given".format(type(A))
    assert isinstance(B, torch.Tensor), "The matrix B must be torch.Tensor, but {} is given".format(type(B))
    assert isinstance(C, torch.Tensor), "The matrix C must be torch.Tensor, but {} is given".format(type(C))

    n, m = B.shape
    A_ = torch.unsqueeze(A, dim=-1)
    B_ = torch.unsqueeze(B, dim=-1)
    C_ = torch.unsqueeze(C, dim=-1)
    Ac = torch.view_as_complex(torch.cat((A_, torch.zeros_like(A_)), dim=-1))
    Bc = torch.view_as_complex(torch.cat((B_, torch.zeros_like(B_)), dim=-1))
    Cc = torch.view_as_complex(torch.cat((C_, torch.zeros_like(C_)), dim=-1))

    def G(s):
        return Cc.mm(torch.inverse(
            s * torch.eye(n) - Ac
        )).mm(Bc)

    def H(g):
        g = g ** 3
        return torch.vstack([
            torch.hstack([torch.hstack([A, g * B.mm(B.T)])]),
            torch.hstack([-g * C.T.mm(C), -A.T])
        ])

    # (a) compute a starting value for gamma_lb using (4.2)
    gamma_lb = compute_gamma_lb(G, A)
    print("initial lb de gamma:", gamma_lb)

    # (b) repeat until 'break'
    while True:
        print("*"*120)
        print("gamma_lb:", gamma_lb)
        # (b1) gamma = (1 + 2 epsilon) * gamma_lb
        gamma = (1 + 2 * epsilon) * gamma_lb
        print("gamma_nw:", gamma)

        # (b2) compute the eigenvalues of H(gamma), (2.2);
        eigs = compute_eig(H(gamma), debug=True)
        freq, _ = torch.sort(eigs.imag)
        # print("pre-freq:", freq)

        # (b3) if no imaginary eigenvalues
        freq = freq[freq > 1e-6]
        print("freq:", freq)
        if len(freq) == 0:
            print("only real eigenvalues")
            gamma_ub = gamma
            break
        # else
        # omega_1 to omega_k = imaginary eigenvalues,
        # m_i = 1/2 * (omega_i + omega_i+1), i = 1, ..., k-1
        ms = 0.5 * (freq[:-1] + freq[1:])
        ms = torch.cat((torch.zeros_like(ms), ms), dim=-1)
        if len(ms.shape) < 2:
            ms = torch.unsqueeze(ms, dim=0)
        print(ms)
        ms = torch.view_as_complex(ms)
        print("after compute next omega", ms)
        # exit()

        # compute the singular vales of G(jm_i)
        sGi = torch.cat([torch.svd(G(m)).S for m in ms])
        print("sing:", sGi)
        gamma_lb = torch.max(sGi)

    # (c) ||G||_inf = 1/2 * (gamma_lb + gamma_ub)
    return 0.5 * (gamma_lb + gamma_ub)


def compute_eig(A, debug=False):
    evals, evecs = np.linalg.eig(A)
    V = evecs.astype(np.complex64)
    V = torch.from_numpy(V)

    A_ = torch.unsqueeze(A, dim=-1)
    Ac = torch.view_as_complex(torch.cat((A_, torch.zeros_like(A_)), dim=-1))
    G = torch.inverse(V).mm(Ac).mm(V)
    G = torch.diagonal(G)
    if debug:
        print("= eigen ========================")
        print("en numpy:\n", evals)
        print("en torch:\n", G)
        print("--------------------------------")
    return G


def compute_gamma_lb(G, A):
    sG0 = torch.max(torch.svd(G(0)).S)

    L = torch.eig(A).eigenvalues
    L = torch.view_as_complex(L)
    L_norms = torch.sqrt(L.real**2+L.imag**2)
    conditions = L.imag / L.real / L_norms
    omega_p = torch.max(conditions)
    sGop = torch.max(torch.svd(G(omega_p)).S)

    sGD = sG0 * 0

    return torch.max(torch.max(sG0, sGop), sGD)
