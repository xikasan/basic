# coding: utf-8

import numpy as np
import torch
from lib.control.gains import *


def test_h_inf():
    A = torch.from_numpy(np.array([[0, 1], [-1, -1]])).float()
    B = torch.from_numpy(np.array([0, 1]).reshape((2, 1))).float()
    C = torch.eye(2).float()

    gamma_G = h_inf(A, B, C)
    print(gamma_G)


def test_eig():
    A = torch.Tensor([[0, -1], [1, 0]]).float()
    eigs = torch.eig(A, eigenvectors=False)
    print(eigs.eigenvalues)


def test_eigendecomposition():
    # A = torch.Tensor([[1, 2], [3, 2]]).float()
    A = torch.Tensor([[1, 2], [-1, -1]]).float()
    eigs = torch.eig(A, eigenvectors=True)
    evals = eigs.eigenvalues
    evecs = eigs.eigenvectors

    eval_norms = torch.square(evals)
    eval_norms = eval_norms[..., 0] - eval_norms[..., 1]
    eval_norms = torch.sqrt(eval_norms)
    eval_norms, eval_indexs = torch.sort(eval_norms)
    evals = evals[eval_indexs, ...]


def test_eigencomposition_with_numpy():
    A = np.array([[1, 2], [-1, -1]]).astype(float)
    print("A:\n", A)
    evals, evecs = np.linalg.eig(A)
    print("vecs:\n", evals)
    print("vecs:\n", evecs)
    V = evecs.copy()

    print("= "*60)
    G = np.linalg.inv(V).dot(A).dot(V)
    G_norm = np.square(G.dot(G.conj()).real)
    G[G_norm < 1e-6] = 0.
    print("come on!!:\n", G)


def test_eigencomposition_with_numpy_torch():
    A = np.array([[1, 2], [-1, -1]]).astype(np.float32)
    evals, evecs = np.linalg.eig(A)
    V = evecs.astype(np.complex64)
    V = torch.from_numpy(V)

    A_ = torch.unsqueeze(torch.from_numpy(A), dim=-1)
    Ac = torch.view_as_complex(torch.cat((A_, torch.zeros_like(A_)), dim=-1))
    G = torch.inverse(V).mm(Ac).mm(V)

    if torch.sum(G.imag ** 2).detach().numpy() == 0.0:
        print("yes real")
        return
    print("in complex")
    evals = torch.diagonal(G)
    print(evals)
    # evals_norm = evals.real ** 2 - evals.imag ** 2
    evals_size = evals.imag
    # evals_norm = torch.dot(evals.conj(), evals)
    print(evals_size)

    # evals, _ = torch.sort(evals)
    # print(evals)


def test_eigencomposition_with_numpy_torch_in_real():
    A = np.array([[7, -6], [3, -2]]).astype(np.float32)
    evals, evecs = np.linalg.eig(A)
    V = evecs.astype(np.complex64)
    V = torch.from_numpy(V)

    A_ = torch.unsqueeze(torch.from_numpy(A), dim=-1)
    Ac = torch.view_as_complex(torch.cat((A_, torch.zeros_like(A_)), dim=-1))
    G = torch.inverse(V).mm(Ac).mm(V)
    print(G)
    print(torch.sum(G.imag).detach().numpy() == 0.)


def singular_value_plot():
    A = np.array([
        [0, 1],
        [-2, -3]
    ]).astype(float)
    B = np.array([0, 5]).reshape((2, 1))
    A = torch.from_numpy(A).float()
    B = torch.from_numpy(B).reshape((2, 1)).float()
    C = torch.eye(2).float()

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

    import xsim
    import pandas as pd
    import matplotlib.pyplot as plt
    log = xsim.Logger()
    omega = 0.01
    for i in range(200):
        s = torch.view_as_complex(
            torch.Tensor([0, omega * i])
        )
        sing = torch.squeeze(torch.svd(G(s)).S)
        sing = sing.detach().numpy()
        log.store(freq=omega*i, singular=sing).flush()

    res = xsim.Retriever(log)
    res = pd.DataFrame(dict(freq=res.freq(), singular=res.singular()))
    print(res)
    print(res[140:150])
    # exit()
    res.plot(x="freq", y="singular")

    plt.savefig("result.png")


def test_in_numpy():
    # sample model
    A = np.array([
        [0, 1],
        [-2, -1]
    ]).astype(float)
    B = np.array([0, 5]).reshape((2, 1)).astype(float)
    C = np.eye(2)
    D = np.zeros_like(B)
    n, m = B.shape
    print("A:\n", A)
    print("B:\n", B)
    print("C:\n", C)
    print("D:\n", D)
    print("pole:\n", np.linalg.eig(A)[0])
    print("="*60)

    def G(s):
        return C.dot(np.linalg.inv(s * np.eye(2) - A)).dot(B) + D

    def H(g):
        g2 = g ** 2
        R = D.T.dot(D) - g2 * np.eye(m)
        S = D.dot(D.T) - g2 * np.eye(n)
        Rinv = np.linalg.inv(R)
        Sinv = np.linalg.inv(S)

        BRCD = B.dot(Rinv).dot(D.T).dot(C)
        ABRDC = A - BRCD
        ACDRB = - A.T + BRCD.T

        gBRB = - g * B.dot(Rinv).dot(B.T)
        gCSC = g * C.T.dot(Sinv).dot(C)
        return np.block([
            [ABRDC, gBRB],
            [gCSC, ACDRB]
        ])

    # initial lower bound de gamma
    sG0 = np.linalg.svd(G(0))[1]
    print(sG0)
    print("-"*60)

    L = np.linalg.eig(A)[0]
    op = [np.abs(l.imag / l.real / np.abs(l)) for l in L]
    op = np.max(op)
    sGo = np.linalg.svd(G(op * 1j))[1]
    print(sGo)
    print("-"*60)
    sGD = np.linalg.svd(D)[1]
    print(sGD)
    print("-"*60)

    gamma_lb = np.max(np.concatenate([sG0, sGo, sGD]))
    print("lower bound de gamma:", gamma_lb)
    print("="*60)

    epsilon = 1e-4
    while True:
        # b1
        print("- b1 ----------")
        gamma = gamma_lb * (1 + epsilon * 2)
        print("gamma:", gamma)
        # b2
        print("- b2 ----------")
        print("H(gamma):\n", H(gamma))
        evals, evecs = np.linalg.eig(H(gamma))
        omegas = np.sort(evals.imag)
        print("all omegas:", omegas)
        omegas = omegas[omegas > 1e-6]
        print("evals:", omegas)

        # b3
        print("- b3 ----------")
        domegas = omegas[1:] - omegas[:-1]
        print("omega gaps:", domegas)
        if len(omegas) == 0 or all([domega < epsilon for domega in domegas]):
            gamma_ub = gamma
            break
        print("omegas_i + omegas_i+1")
        print(omegas[1:], omegas[:-1])
        ms = 0.5 * (omegas[1:] + omegas[:-1]) * 1j
        print("ms:", ms)

        sigs = [np.max(np.linalg.svd(G(m))[1]) for m in ms]
        gamma_lb = np.max(sigs)
        print("gamma", gamma_lb)

    gain = 0.5 * (gamma_lb + gamma_ub)
    print("Hinf gain:", gain)
    exit()

    import xsim
    import pandas as pd
    import matplotlib.pyplot as plt
    log = xsim.Logger()
    omega = 1
    for i in range(500):
        s = omega * i * 1.j
        sing = np.linalg.svd(G(s))[1]
        print(sing)
        log.store(freq=omega * i, singular=sing).flush()
    #
    res = xsim.Retriever(log)
    res = pd.DataFrame(dict(freq=res.freq(), singular=res.singular()))
    res.plot(x="freq", y="singular")

    plt.savefig("result-numpy.png")


if __name__ == '__main__':
    test_in_numpy()
    # test_h_inf()
    # test_eig()
    # test_eigendecomposition()
    # test_eigencomposition_with_numpy()
    # test_eigencomposition_with_numpy_torch()
    # test_eigencomposition_with_numpy_torch_in_real()
    # singular_value_plot()
