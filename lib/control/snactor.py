# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Union, List
from lib.utils.norm import spectral_norm as SN


class SNActor(nn.Module):

    def __init__(
            self,
            dim_state: int,
            dim_action: int,
            units: Union[int] = (32, 32),
            bias: bool = False,
            norm_scale: float = 1.0
    ):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(dim_state, units[0], bias=bias))
        self.fc2 = spectral_norm(nn.Linear(units[0], units[1], bias=bias))
        self.fc3 = nn.Linear(units[1], dim_action, bias=bias)
        self.delta = norm_scale

    def forward(self, x):
        x = self.fc1(x) * self.delta
        x = torch.tanh(x)
        x = self.fc2(x) * self.delta
        x = torch.tanh(x)
        mu = self.fc3(x)
        std = torch.ones_like(mu)
        return mu, std

    def spectral_norms(self):
        return [
            SN(self.fc1.weight) * self.delta,
            SN(self.fc2.weight) * self.delta,
            SN(self.fc3.weight)
        ]


class SFB_SNActor(nn.Module):

    def __init__(
            self,
            dim_state: int,
            dim_action: int,
            units: List[int] = (32, 32),
            bias: bool = False,
            norm_scale: float = 1.0,
            init_K: np.ndarray = None
    ):
        super().__init__()
        self.pi = SNActor(dim_state, dim_action, units, bias, norm_scale)
        self.fb_gain = nn.Linear(dim_state, dim_action, bias=False)
        if init_K is not None:
            if isinstance(init_K, np.ndarray):
                init_K = torch.from_numpy(init_K).float()
            self.fb_gain.weight.data.copy_(init_K.reshape(self.K.shape))

    def forward(self, x):
        mu, std = self.pi(x)
        ufb = self.fb_gain(x)
        return mu + ufb, mu, std

    @property
    def K(self):
        return self.fb_gain.weight.data

    def spectral_norms(self):
        return self.pi.spectral_norms()
