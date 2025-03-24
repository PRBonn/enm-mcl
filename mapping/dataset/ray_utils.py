"""The functions for supporting the LiDARDataset dataset class
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved

Code partially borrowed from
https://github.com/PRBonn/ir-mcl/blob/main/nof/dataset/ray_utils.py
MIT License
Copyright (c) 2023 Haofei Kuang, Xieyuanli Chen, Tiziano Guadagnino,
Nicky Zimmerman, Jens Behley, Cyrill Stachniss
"""

import torch


def get_ray_directions(angle_min, angle_max, angle_res):
    """
    Get the direction of laser beam in lidar coordinate.

    :param angle_min: the start angle of one scan (scalar)
    :param angle_max: the end angle of one scan (scalar)
    :param angle_res: the interval angle between two consecutive beams (scalar)

    :return: the direction of each beams (shape: (N, 2))
    """
    r = 1
    beams = torch.arange(angle_min, angle_max, angle_res)

    x = r * torch.cos(beams)
    y = r * torch.sin(beams)

    directions = torch.stack([x, y], dim=-1)
    return directions


def get_rays(directions, T_w2l):
    """
    Get ray origin and normalized directions in world coordinate for all beams in one scan.

    :param directions: ray directions in the lidar coordinate. (shape: (N, 2))
    :param T_w2l: 2*3 transformation matrix from lidar coordinate
                  to world coordinate. (shape: (2, 3))

    :return rays_0: the origin of the rays in world coordinate. (shape: (N, 2))
    :return rays_d: the normalized direction of the rays in world coordinate. (shape: (N, 2))

    """
    rays_d = directions @ T_w2l[:, :2].T  # (N, 2)

    # normalize direction vector
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = T_w2l[:, 2].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 2)
    rays_o = rays_o.view(-1, 2)

    return rays_o, rays_d
