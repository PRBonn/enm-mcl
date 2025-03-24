"""The script for display the information of the neural map model
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved
"""
import argparse
import yaml
import torch

from mapping.map import MapRepresentation


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='enm-mcl/configs/mapping_enm.yaml',
                        help='the path for the configuration file.')

    return parser.parse_args()


def eval(cfg):
    print("Initializing the mapping process......")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # model
    if cfg['map_type'] == 'ENM':
        map_representation = MapRepresentation(cfg, device)
    else:
        raise ValueError("Unknown map type: {}".format(cfg['map_type']))
    map_representation.load(cfg['results_path'])
    print("\nMapping process initialized! Start evaluating the map model......")

    # evaluation
    print("Evaluating the map model......")
    print("Done!\n")

    # Memory Cost
    print("Compute map cost......")
    memory_cost = map_representation.cal_memory_cost()
    print("Memory Cost is {} MB".format(memory_cost))
    print("Done!\n")

    # visualization
    print("Visualizing the map model......")
    if cfg['map_type'] == 'ENM':
        map_representation.plot_map(cfg['map_size'], 0.05)
    else:
        raise ValueError("Unknown map type: {}".format(cfg['map_type']))
    print("Done!\n")

    print("Mapping process finished......")


if __name__ == '__main__':
    args = get_args()
    # load config file
    config_filename = args.config_file
    cfg = yaml.safe_load(open(config_filename))

    eval(cfg=cfg)
