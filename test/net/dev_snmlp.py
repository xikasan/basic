# coding: utf-8

import numpy as np
import torch
from lib.net.snmlp import SNMLP


def run():
    dim_input = 3
    dim_output = 1
    hidden_size = (32, 32)
    net = SNMLP(dim_input, dim_output, hidden_size)
    print(net)

    x = np.random.random((5, dim_input))
    x = torch.from_numpy(x).float()
    print("dummy input:\n", x)
    mu, std = net(x)
    print("mu:\n", mu)
    print("std:\n", std)


if __name__ == '__main__':
    run()
