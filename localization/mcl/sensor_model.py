"""The class for the ENM-based observation model
author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved.
"""
import numpy as np
import torch

from mapping.map import MapRepresentation


class SensorModel:
    def __init__(self, scans, cfg_loc, cfg_map):
        # load the map module.
        self.scans = scans

        use_cuda: bool = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        # self.device = torch.device('cpu')
        print('Using: ', self.device)

        # load lidar
        self.directions = cfg_loc['directions'].to(self.device)
        self.obs_area = cfg_loc['obs_area']
        self.lambda_avg = cfg_loc['lambda_avg']

        # load sensor model
        self.sensor_model_type = cfg_map['map_type']
        if self.sensor_model_type == 'ENM':
            self.map_representation = MapRepresentation(cfg_map, self.device)
        else:
            raise "Not supported sensor model: {}".format(self.sensor_model_type)

        self.map_representation.load(cfg_map['results_path'])

    def get_beam_end_points(self, Ts_w2l, z_obs):
        """
        get the beam end points of each particle
        :param Ts_w2l: (N, 3, 3)
        :return: (N, N_ray, 2)
        """
        # ndarray to tensor
        Ts_w2l = torch.from_numpy(Ts_w2l).float().to(self.device)
        z_obs = torch.from_numpy(z_obs).float().to(self.device)

        # get ray directions
        rays_d = torch.einsum('nij, kj->nki', Ts_w2l[:, :2, :2], self.directions)
        # normalize direction vector
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        # ranges2points (in world frame)
        # n: N_particles, i: N_rays, j: point's dimension
        points = torch.einsum('i, ij->ij', z_obs, self.directions)
        points = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
        points = torch.einsum('nkj, ji->nki', Ts_w2l, points.T)[:, :2]
        points = points.transpose(2, 1)  # (N, N_ray, 2)

        return points, rays_d

    def update_weights(self, particles, frame_idx, T_b2l=None):
        current_scan = self.scans[frame_idx]

        particle_poses = particles[..., :-1]
        # 1. particles_poses (N, 3) to particles_mat (N, 4, 4)
        xs, ys, yaws = particle_poses[:, 0], particle_poses[:, 1], particle_poses[:, 2]
        particles_mat = [[np.cos(yaws), -np.sin(yaws), np.zeros_like(xs), xs],
                         [np.sin(yaws), np.cos(yaws), np.zeros_like(xs), ys],
                         [np.zeros_like(xs), np.zeros_like(xs), np.ones_like(xs), np.zeros_like(xs)],
                         [np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.ones_like(xs)]]
        particles_mat = np.array(particles_mat).transpose((2, 0, 1))

        # 2. transfer robot poses to lidar poses: T_w2b @ T_b2l -> T_w2l
        if T_b2l is not None:
            particles_mat = np.einsum('nij, jk->nik', particles_mat, T_b2l)
        particles_mat = particles_mat[:, [0, 1, 3]][:, :, [0, 1, 3]]  # shape: (N, 3, 3)

        # update weights with the BeamEnd model
        # 3. compute the BeamEnd points of each particle
        beam_end_points, beam_dirs = self.get_beam_end_points(particles_mat, current_scan)

        # 4. compute the likelihood of each BeamEnd point via Map Representation
        N_particles, N_rays = beam_end_points.shape[:2]
        beam_end_points = beam_end_points.reshape((N_particles * N_rays, 2))
        beam_dirs = beam_dirs.reshape((N_particles * N_rays, 2))

        sdf_values, pdf_values = self.map_representation.predict(beam_end_points, directions=beam_dirs)
        sdf_values = torch.abs(sdf_values)
        pdf_values = torch.abs(pdf_values)

        average_values = (sdf_values + pdf_values) / 2
        average_values = torch.abs(average_values)
        average_values = average_values.reshape((N_particles, N_rays))

        scores = observation_score(average_values, self.lambda_avg, current_scan, obs_area=self.obs_area)

        # update the particles' weight
        scores = scores.cpu().numpy()
        particles[:, 3] = particles[:, 3] * scores

        # normalize the particles' weight
        particles[:, 3] /= np.sum(particles[:, 3])

        return particles


def observation_score(obs, lambda_obs, current_scan, obs_area):
    N_particles, N_rays = obs.shape[:2]
    step = N_rays // obs_area
    scores = torch.ones((N_particles,)).float().to(obs.device)
    for indices in range(0, N_rays, step):
        # only update the valid beams
        if np.sum(current_scan[indices:indices + step] > 0):
            scores *= torch.exp(
                -torch.mean(
                    lambda_obs * obs[:, indices:indices + step][:, current_scan[indices:indices + step] > 0],
                    dim=1)
            )

    return scores
