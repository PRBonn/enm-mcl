"""The class MapRepresentation is used for 2D mapping using Efficient Neural Map.
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved
"""
import os.path
import sys

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import optim
import torch.nn as nn

from .models.models import EfficientNeuralMap


class MapRepresentation:
    """
    Efficient Neural Map for 2D Mapping

    Args:
        cfg: dict, configuration parameters
        device: torch.device, device to run
    """
    def __init__(self, cfg, device):
        self.device = device
        self.map_size = cfg['map_size']

        self.model = EfficientNeuralMap(
            map_size=cfg['map_size'], grid_res=cfg['grid_res'], level_num=cfg['level_num'],
            up_scale_ratio=cfg['up_scale_ratio'], feature_dim=cfg['feature_dim'],
            hidden_dim=cfg['hidden_dim'], with_bias=cfg['with_bias'],
            use_dir=cfg['use_dir'], pe_freq=cfg['pe_freq'],
        )
        self.model.to(self.device)

        # ray sampling parameters
        self.truncated_area = cfg['truncated_area']
        self.truncated_samples_num = cfg['truncated_samples_num']
        self.occupied_area = cfg['occupied_area']
        self.occupied_sampled_num = cfg['occupied_sampled_num']
        self.free_space_sampled_num = cfg['free_space_sampled_num']

        self.scalar_factor = -5.0 / self.truncated_area

        # define optimization parameters
        params = list(self.model.parameters())
        lr = cfg['learning_rate']
        self.optimizer = optim.Adam(params, lr, betas=(0.9, 0.99), eps=1e-15)
        self.loss_sdf = nn.BCELoss()
        self.loss_pdf = nn.SmoothL1Loss()

    def fit(self, observation):
        """
        Args:
            observation: (origins: (batch_size, 2), endpoints: (batch_size, 2))
        Returns:
            loss: float, loss value
        """
        self.model.train()

        origins, ray_dirs, endpoints = observation
        surface_data, surface_gts, free_data, free_gts = self.ray_sample(endpoints, origins)

        truncated_dirs = ray_dirs.repeat_interleave(self.truncated_samples_num, dim=0)
        occupied_dirs = ray_dirs.repeat_interleave(self.occupied_sampled_num, dim=0)
        input_dir = torch.cat([truncated_dirs, occupied_dirs], dim=0)

        predict_surface_sdf, predict_surface_pdf, _, = self.model(surface_data, input_dir)
        predict_free_sdf, _, _, = self.model(free_data)

        predict_sdf = torch.cat([predict_surface_sdf, predict_free_sdf])

        g = self.get_numerical_gradient(
            surface_data,
            predict_sdf,
            self.model.grid_res * 0.2,  # 0.5, 0.4, 0.2
        )
        eikonal_loss = ((1.0 - g.norm(2, dim=-1)) ** 2).mean()  # MSE with regards to 1

        predict_sdf = torch.sigmoid(predict_sdf * self.scalar_factor)
        traget_surface_sdf = torch.sigmoid(surface_gts * self.scalar_factor)
        target_sdf = torch.cat([traget_surface_sdf, free_gts], dim=0)
        loss_sdf = self.loss_sdf(predict_sdf, target_sdf) + 0.1 * eikonal_loss

        loss_pdf = self.loss_pdf(predict_surface_pdf, surface_gts)
        loss = loss_sdf + loss_pdf

        # print(loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def predict(self, positions, directions=None):
        """
        Args:
            positions: (batch_size, 2)
        Returns:
            predict_sdf: (batch_size)
        """
        self.model.eval()
        with torch.no_grad():
            if directions is not None:
                predict_sdf, predict_pdf, mask = self.model(positions, directions)
                predict_sdf[mask] = 5.0
                predict_pdf[mask] = 5.0
                return predict_sdf, predict_pdf
            else:
                predict_sdf, _, mask = self.model(positions)
                predict_sdf[mask] = 5.0

                return predict_sdf

    def ray_sample(self, points, translation):
        shift_points = points - translation
        n = shift_points.shape[0]
        pdim = shift_points.shape[1]
        truncated_samples = torch.rand(n * self.truncated_samples_num, 1, device=self.device) * self.truncated_area
        occupied_sample = -torch.rand(n * self.occupied_sampled_num, 1, device=self.device) * self.occupied_area

        distances = torch.norm(shift_points, p=2, dim=1, keepdim=True)
        repeated_truncated_distances = distances.repeat(1, self.truncated_samples_num).reshape(-1, 1)
        repeated_occupied_distances = distances.repeat(1, self.occupied_sampled_num).reshape(-1, 1)

        truncated_ratios = 1.0 - truncated_samples / repeated_truncated_distances
        occupied_ratios = 1.0 - occupied_sample / repeated_occupied_distances
        repeated_free_ratios = 1 - self.truncated_area / (
            distances.repeat(1, self.free_space_sampled_num).reshape(-1, 1))
        free_space_ratios = torch.rand(n * self.free_space_sampled_num, 1, device=self.device) * repeated_free_ratios

        truncated_ratios = truncated_ratios.reshape(n, -1)
        occupied_ratios = occupied_ratios.reshape(n, -1)

        truncated_pd = truncated_samples.reshape(n, -1)
        occupied_pd = occupied_sample.reshape(n, -1)
        free_space_occupancy = torch.zeros(n * self.free_space_sampled_num, device=self.device)

        surface_ratios = torch.cat((truncated_ratios, occupied_ratios), 1).reshape(-1, 1)
        surface_pd = torch.cat((truncated_pd, occupied_pd), 1).reshape(-1, 1)
        surface_r_points = shift_points.repeat(1, self.truncated_samples_num + self.occupied_sampled_num).reshape(-1,
                                                                                                                  pdim)
        surface_r_translations = translation.repeat(1, self.truncated_samples_num + self.occupied_sampled_num).reshape(
            -1, pdim)

        surface_sample_points = surface_r_points * surface_ratios + surface_r_translations

        free_r_points = shift_points.repeat(1, self.free_space_sampled_num).reshape(-1, pdim)
        free_r_translations = translation.repeat(1, self.free_space_sampled_num).reshape(-1, pdim)
        free_sample_points = free_r_points * free_space_ratios + free_r_translations

        return surface_sample_points, surface_pd.squeeze(1), free_sample_points, free_space_occupancy

    def get_numerical_gradient(self, x, sdf_x=None, eps=0.02, two_side=True):

        N = x.shape[0]

        eps_x = torch.tensor([eps, 0.0], dtype=x.dtype, device=x.device)  # [2]
        eps_y = torch.tensor([0.0, eps], dtype=x.dtype, device=x.device)  # [2]

        if two_side:
            x_pos = x + eps_x
            x_neg = x - eps_x
            y_pos = x + eps_y
            y_neg = x - eps_y

            x_posneg = torch.concat((x_pos, x_neg, y_pos, y_neg), dim=0)
            sdf_x_posneg = self.model(x_posneg)[0].unsqueeze(-1)

            sdf_x_pos = sdf_x_posneg[:N]
            sdf_x_neg = sdf_x_posneg[N: 2 * N]
            sdf_y_pos = sdf_x_posneg[2 * N: 3 * N]
            sdf_y_neg = sdf_x_posneg[3 * N:]

            gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
            gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)

        else:
            x_pos = x + eps_x
            y_pos = x + eps_y

            x_all = torch.concat((x_pos, y_pos), dim=0)
            sdf_x_all = self.model(x_all)[0].unsqueeze(-1)

            sdf_x = sdf_x.unsqueeze(-1)

            sdf_x_pos = sdf_x_all[:N]
            sdf_y_pos = sdf_x_all[N: 2 * N]

            gradient_x = (sdf_x_pos - sdf_x) / eps
            gradient_y = (sdf_y_pos - sdf_x) / eps

        gradient = torch.cat([gradient_x, gradient_y], dim=1)  # [...,2]

        return gradient

    def plot_map(self, map_size, map_res):
        min_x, max_x, min_y, max_y = map_size
        x = torch.arange(round((max_x - min_x) / map_res) + 1, dtype=torch.long)
        y = torch.arange(round((max_y - min_y) / map_res) + 1, dtype=torch.long)

        sample_x, sample_y = torch.meshgrid(x, y)
        coord_xy = torch.stack((sample_x.flatten(), sample_y.flatten())).float()

        coord_xy *= map_res
        coord_xy[0, :] += min_x
        coord_xy[1, :] += min_y

        coord_xy = coord_xy.T.cuda()
        output = self.predict(coord_xy)
        grid_shape = sample_x.shape
        sdf_2d = output.reshape(grid_shape)

        # Create a figure with a colorbar
        fig, ax = plt.subplots()
        im = ax.imshow(sdf_2d.cpu().numpy().T, cmap='jet', origin='lower', extent=map_size, vmin=-0.1, vmax=0.5)

        # use make_axes_locatable to create a colorbar with the same height as ax
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("SDF Value (m)")

        ax.set_xlim(-15, 17)
        ax.set_ylim(-12, 4)

        ax.axis("off")
        plt.savefig("sdf_map.pdf", format='pdf', bbox_inches='tight', dpi=1200, pad_inches=0)
        plt.show()

        return sdf_2d

    def cal_memory_cost(self):
        total_params_memory = 0
        for name, param in self.model.named_parameters():
            total_params_memory += sys.getsizeof(param.data.cpu().numpy())
        total_params_memory /= 1024
        return total_params_memory

    def save(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state_dict, path)
        print(f"Model and optimizer saved to {path}")

    def load(self, path):
        state_dict = torch.load(path, weights_only=False)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        print(f"Model and optimizer loaded from {path}")
