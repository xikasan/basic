# coding: utf-8

import numpy as np
import torch
from lib.control.gains import *


def test_in_numpy():
    # sample model
    A = np.array([
        [0, 1],
        [-10, -1]
    ]).astype(float)
    B = np.array([0, 1]).reshape((2, 1)).astype(float)
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
    print(np.linalg.eig(A))
    exit()
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


def dev_H_inf():
    # sample model
    A = np.array([
        [0, 1],
        [-10, -1]
    ]).astype(float)
    B = np.array([0, 1]).reshape((2, 1)).astype(float)
    C = np.eye(2)

    tcA = torch.from_numpy(A)
    tcB = torch.from_numpy(B)
    tcC = torch.from_numpy(C)

    gamma = h_inf(tcA, tcB, tcC)
    print("="*60)
    print("Hinf gain:", gamma)


if __name__ == '__main__':
    dev_H_inf()
    # test_in_numpy()
