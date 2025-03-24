"""The class of Efficient Neural Map
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved
"""

import time

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PENeRF(nn.Module):
    """
    Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)

    Args:
        in_channels: int, the number of input channels (2 for both xyz and direction)
        N_freq: int, the number of frequency bands
        logscale: bool, whether to use logscale frequency bands (default: True)
    """

    def __init__(self, in_channels, N_freq, logscale=True):
        super(PENeRF, self).__init__()
        self.N_freq = N_freq
        self.in_channels = in_channels

        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freq - 1, N_freq)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freq - 1), N_freq)

        self.pe_dim = in_channels * N_freq * 2 + 2

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)

        Args:
            x: (B, self.in_channels)
        Return:
            out: (B, self.N_freq * self.in_channels * len(self.funcs))
        """
        pos_embedded = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                pos_embedded.append(func(freq * x))

        pos_embedded = torch.cat(pos_embedded, -1)
        return pos_embedded


class EfficientNeuralMap(nn.Module):
    """
    Efficient Neural Map for 2D SDF and Projective-SDF prediction

    Args:
        map_size: list, the size of the feature grids
        grid_res: float, the resolution of the feature grids
        level_num: int, the number of feature grids
        up_scale_ratio: float, the up scale ratio of the feature grids
        feature_dim: int, the dimension of the feature
        hidden_dim: int, the hidden dimension of the feature encoder
        with_bias: bool, whether to use bias in the feature encoder
        use_dir: bool, whether to use directional encoding
        pe_freq: int, the frequency of positional encoding
    """
    def __init__(self, map_size=[-25, 25, -25, 25], grid_res=0.1,
                 level_num=1, up_scale_ratio=1.5, feature_dim=8,
                 hidden_dim=64, with_bias=True, use_dir=True, pe_freq=4):
        super(EfficientNeuralMap, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.with_bias = with_bias

        # feature grids
        self.map_size = map_size
        self.grid_res = grid_res
        self.level_num = level_num
        self.up_scale_ratio = up_scale_ratio

        self.grids_position = None
        self.grids_features = nn.ParameterList([])
        self.__init_grid()

        # grids feature encoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.feature_dim, out_features=self.hidden_dim, bias=self.with_bias),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=self.with_bias),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=self.with_bias),
            nn.ReLU(),
        )

        # directional encoder
        self.use_dir = use_dir
        self.pos_encoder = PENeRF(in_channels=2, N_freq=pe_freq)
        dir_hidden_dim = self.hidden_dim + self.pos_encoder.pe_dim
        self.directional_encoder = nn.Sequential(
            nn.Linear(in_features=dir_hidden_dim, out_features=dir_hidden_dim, bias=self.with_bias),
            nn.ReLU(),
            nn.Linear(in_features=dir_hidden_dim, out_features=dir_hidden_dim, bias=self.with_bias),
            nn.ReLU(),
            nn.Linear(in_features=dir_hidden_dim, out_features=dir_hidden_dim, bias=self.with_bias),
            nn.ReLU(),
        )

        self.sdf_header = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, self.with_bias),
        )

        self.pdf_header = nn.Sequential(
            nn.Linear(dir_hidden_dim, 1, self.with_bias),
        )

    def __init_grid(self):
        min_x, max_x, min_y, max_y = self.map_size

        t = time.time()
        for i in range(self.level_num):
            current_grid_res = self.grid_res * (self.up_scale_ratio ** i)
            x_steps = math.ceil((max_x - min_x) / current_grid_res) + 1
            y_steps = math.ceil((max_y - min_y) / current_grid_res) + 1
            grids_feature = nn.Parameter(
                torch.zeros(size=(1, self.feature_dim, y_steps, x_steps))
            )
            self.grids_features.append(grids_feature)

        print("Feature grids initialization finished! Time consume: {:.2f}s".format(time.time() - t))

    def __get_grid_features(self, x):
        B, _ = x.shape
        min_x, max_x, min_y, max_y = self.map_size
        x_length = max_x - min_x
        y_length = max_y - min_y

        sum_features = torch.zeros(x.shape[0], self.feature_dim, device=x.device)
        for i in range(self.level_num):
            normalize_coords = torch.zeros_like(x)
            normalize_coords[:, 0] = ((x[:, 0] - min_x) / x_length) * 2 - 1
            normalize_coords[:, 1] = ((x[:, 1] - min_y) / y_length) * 2 - 1
            normalize_coords = normalize_coords.reshape(1, B, 1, 2)

            # get features
            sum_features += F.grid_sample(
                self.grids_features[i], normalize_coords, mode='bilinear',
                align_corners=True, padding_mode='zeros')[0, :, :, 0].transpose(0, 1)

        return sum_features

    def forward(self, x, d=None):
        # get grid features by bilinear interpolation
        features = self.__get_grid_features(x)
        features = F.layer_norm(features, normalized_shape=[self.feature_dim])

        mask = (torch.sum(features**2, dim=1) == 0)

        # grids feature encoding
        features = self.decoder(features)
        sdf = self.sdf_header(features).squeeze(dim=1)

        if d is None:
            return sdf, None, mask

        # directional encoding
        pe_dir = self.pos_encoder(d)
        features = torch.cat([features, pe_dir], dim=1)
        features = self.directional_encoder(features)
        pdf = self.pdf_header(features).squeeze(dim=1)

        return sdf, pdf, mask
