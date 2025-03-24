"""This file defines the dataset class for the LiDAR dataset.
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved
"""
import os
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
import torch.utils.data as data

from .ray_utils import get_ray_directions, get_rays


class BeamDataset(data.Dataset):
    def __init__(self, ray_origins, ray_dirs, ray_endpoints):
        super(BeamDataset, self).__init__()
        self.ray_origins = ray_origins
        self.ray_dirs = ray_dirs
        self.ray_endpoints = ray_endpoints

        assert len(self.ray_origins) == len(ray_dirs) == len(ray_endpoints), \
            "The lengths of ray_origins, ray_dirs, ray_endpoints are not equal."

    def __len__(self):
        return len(self.ray_origins)

    def __getitem__(self, index):
        ray_origin = self.ray_origins[index]
        ray_dir = self.ray_dirs[index]
        ray_endpoint = self.ray_endpoints[index]

        return ray_origin, ray_dir, ray_endpoint


class LiDARDataset(data.Dataset):
    def __init__(self, root_dir, split='train', step=1):
        super(LiDARDataset, self).__init__()
        self.root_dir = root_dir

        assert split in ['train', 'val', 'test'], \
            "Not supported split type \"{}\"".format(split)
        self.split = split

        self.step = step
        self.positions = []
        self.ray_directions = []
        self.all_points = []
        self.load_data()

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = self.positions[index]
        ray_dir = self.ray_directions[index]
        points = self.all_points[index]

        return position, ray_dir, points

    def load_data(self):
        t = time.time()
        print("Loading dataset......")
        file_path = os.path.join(self.root_dir, '{}.json'.format(self.split))

        with open(file_path, 'r') as f:
            self.meta = json.load(f)

        self.num_beams = self.meta['num_beams']
        self.angle_min = self.meta['angle_min']
        self.angle_max = self.meta['angle_max']
        self.angle_res = self.meta['angle_res']
        self.max_range = self.meta['max_range']
        self.meta['scans'] = self.meta['scans']

        # ray directions for all beams in the lidar coordinate, shape: (N, 2)
        self.directions = get_ray_directions(self.angle_min, self.angle_max, self.angle_res)

        for i in range(0, len(self.meta['scans']), self.step):
            scan = self.meta['scans'][i]
            pose = scan['pose_gt']

            # convert [x,y, yaw] to transformation matrix
            translation = pose[:2]
            rotation = R.from_euler('z', pose[2], degrees=False).as_matrix()[:2, :2]
            pose = np.eye(3)
            pose[:2, :2] = rotation
            pose[:2, 2] = translation
            pose = pose[:2, :3]
            T_w2l = torch.FloatTensor(pose)

            rays_o, rays_d = get_rays(self.directions, T_w2l)
            range_readings, valid_mask_gt = self._load_scan(scan['range_reading'])

            # ranges to point cloud
            points = rays_o + rays_d * range_readings.unsqueeze(-1)

            position = T_w2l[:, 2]
            rays_d = rays_d[valid_mask_gt]
            points = points[valid_mask_gt]

            # save the data
            self.positions.append(position)
            self.ray_directions.append(rays_d)
            self.all_points.append(points)

        print("Loading dataset......done in {:.2f}s".format(time.time() - t))

    def _load_scan(self, range_readings):
        range_readings = np.array(range_readings)

        # valid mask ground truth (< max_range)
        valid_mask_gt = range_readings.copy()
        valid_mask_gt[np.logical_and(valid_mask_gt > 0, valid_mask_gt < self.max_range)] = 1
        valid_mask_gt[valid_mask_gt >= self.max_range] = 0

        # set invalid value (no return) to 0
        range_readings[range_readings >= self.max_range] = 0

        range_readings = torch.Tensor(range_readings)
        valid_mask_gt = torch.BoolTensor(valid_mask_gt)

        return range_readings, valid_mask_gt
