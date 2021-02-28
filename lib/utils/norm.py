# coding: utf-8

import numpy as np
import torch


def spectral_norm(x):
    if isinstance(x, np.ndarray):
        return np.max(np.linalg.svd(x)[1])
    return torch.max(torch.svd(x).S)

