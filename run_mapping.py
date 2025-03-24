"""The main function for training the neural map model
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved
"""

import argparse
import yaml
from tqdm import tqdm

import torch

from mapping.dataset.lidardataset import LiDARDataset
from mapping.map import MapRepresentation


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='enm-mcl/configs/mapping_enm.yaml',
                        help='the path for the configuration file.')

    return parser.parse_args()


def run(cfg):
    print("Initializing the mapping process......")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # dataset
    dataset = LiDARDataset(root_dir=cfg['root_dir'], split=cfg['split'], step=cfg['step'])
    print("\nDataset loaded! Total number of frames: ", len(dataset))

    # model
    if cfg['map_type'] == 'ENM':
        map_representation = MapRepresentation(cfg, device)
    else:
        raise ValueError("Unknown map type: {}".format(cfg['map_type']))

    print("\nMapping process initialized! "
          "Map representation is: {}".format(cfg['map_type']))

    # run mapping
    print("\nStart mapping process......")
    if cfg['map_type'] == 'ENM':
        ray_endpoints = torch.FloatTensor([]).to(device)
        ray_dirs = torch.FloatTensor([]).to(device)
        ray_origins = torch.FloatTensor([]).to(device)
        for position, ray_dir, points in dataset:
            points = points.to(device)
            ray_dir = ray_dir.to(device)
            position = position.to(device)

            ray_endpoints = torch.cat((ray_endpoints, points), 0).float()
            ray_dirs = torch.cat((ray_dirs, ray_dir), 0).float()
            ray_origins = torch.cat((ray_origins, position.repeat(points.shape[0], 1)), 0).float()

        iterations = cfg['iterations']
        batch_size = cfg['batch_size']

        with tqdm(total=iterations) as pbar:
            pbar.set_description('map Processing:')
            for itr in range(iterations):
                ray_indices = torch.randint(0, ray_endpoints.shape[0], (batch_size,), device=device)
                endpoints_sample = ray_endpoints[ray_indices]
                origins_sample = ray_origins[ray_indices]
                ray_dirs_sample = ray_dirs[ray_indices]

                observations = (origins_sample, ray_dirs_sample, endpoints_sample)
                batch_loss = map_representation.fit(observations)

                pbar.update(1)
                pbar.set_postfix(loss='{:.4f}'.format(batch_loss))
    else:
        raise ValueError("Unknown map type: {}".format(cfg['map_type']))

    # save the map
    map_representation.save(cfg['results_path'])

    print("\nMapping process finished! The map is saved in {}.".format(cfg['results_path']))


if __name__ == '__main__':
    args = get_args()
    # load config file
    config_filename = args.config_file
    cfg = yaml.safe_load(open(config_filename))

    run(cfg=cfg)
