"""The main function for global localization experiments
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved

Code partially borrowed from
https://github.com/PRBonn/ir-mcl/blob/main/main.py.
MIT License
Copyright (c) 2023 Haofei Kuang, Xieyuanli Chen, Tiziano Guadagnino,
Nicky Zimmerman, Jens Behley, Cyrill Stachniss
"""

import argparse
import os
import random

import torch
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from localization.utils import load_data, summary_loc
from localization.mcl.initialization import init_particles_pose_tracking, init_particles_uniform
from localization.mcl.motion_model import gen_commands_srrg
from localization.mcl.sensor_model import SensorModel
from localization.mcl.srrg_utils.pf_library.pf_utils import PfUtils
from localization.mcl.vis_loc_result import plot_traj_result
from localization.mcl.visualizer import Visualizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str,
                        default='enm-mcl/configs/global_localization/loc_config_test1.yaml',
                        help='the path for the configuration file.')
    parser.add_argument('--suffix', type=str, default='', help='the suffix for the result file.')
    parser.add_argument('--seed', type=int, default=42, help='random seed for the experiment.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print("=============================================")
    print("Experiment ID: ", args.suffix)
    print("Random seed: ", args.seed)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load config file
    config_filename = args.config_file
    cfg_loc = yaml.safe_load(open(config_filename))

    # load parameters
    start_idx = cfg_loc['start_idx']
    grid_res = cfg_loc['grid_res']
    numParticles = cfg_loc['numParticles']
    # numParticles = 100000
    reduced_num = cfg_loc['num_reduced']
    visualize = cfg_loc['visualize']
    result_path = cfg_loc['result_path']

    # load input data
    map_pose_file = cfg_loc['map_pose_file']
    data_file = cfg_loc['data_file']
    mapsize = cfg_loc['map_size']

    print('\nLoading data......')
    t = time.time()
    # load poses
    timestamps_mapping, map_poses, _, _, _ = load_data(map_pose_file)
    timestamps_gt, poses_gt, odoms, scans, params = \
        load_data(data_file, max_beams=cfg_loc['max_beams'])
    cfg_loc.update(params)

    if cfg_loc['T_b2l']:
        T_b2l = np.loadtxt(cfg_loc['T_b2l'])
    else:
        T_b2l = None

    # loading the occupancy grid map for visualization
    occmap = np.load(cfg_loc['map_file'])

    print('All data are loaded! Time consume: {:.2f}s'.format(time.time() - t))

    # initialize sensor model
    # load parameters of sensor model
    cfg_map = yaml.safe_load(open(cfg_loc['map_config']))
    print("\nSetting up the Sensor Model......")
    sensor_model = SensorModel(scans, cfg_loc, cfg_map)
    print("Sensor Model Setup Successfully!\n")

    # initialize particles
    print('\nMonte Carlo localization initializing...')
    # map_size, road_coords = gen_coords_given_poses(map_poses)
    if cfg_loc['pose_tracking']:
        init_noise = cfg_loc['init_noise']
        particles = init_particles_pose_tracking(
            numParticles, poses_gt[start_idx], noises=init_noise)
    else:
        # particles = init_particles_uniform(mapsize, numParticles)
        particles = init_particles_uniform(
            mapsize, numParticles, sensor_model.map_representation,
            threshold_free=cfg_map['threshold_free'],
            threshold_unknown=cfg_map['threshold_unknown'])

    print("\nAfter initialization, the number of particles is: ", particles.shape[0])

    # generate odom commands
    commands = gen_commands_srrg(odoms)
    srrg_utils = PfUtils()

    # initialize a visualizer
    if visualize:
        plt.ion()
        visualizer = Visualizer(mapsize, poses=poses_gt, map_poses=map_poses, occ_map=occmap,
                                odoms=odoms, grid_res=grid_res, start_idx=start_idx)

    # Starts MCL
    results = np.full((len(poses_gt), numParticles, 4), 0, np.float32)
    # add a small offset to avoid showing the meaning less estimations before convergence
    offset = 0
    # for checking converge
    moving_dist = 0.0
    is_converged = False
    converge_time_threshold = 0
    valid_frames = []

    # for time cost
    pre_converged_times = []
    post_converged_times = []
    total_times = []

    # tqdm progress bar, using frame_idx as the iterator
    progress_bar = tqdm(range(start_idx, len(poses_gt)), desc="Processing frames")
    for frame_idx in progress_bar:
        curr_timestamp = timestamps_gt[frame_idx]
        start = time.time()

        # motion model
        particles = srrg_utils.motion_model(particles, commands[frame_idx])

        dist = np.linalg.norm(commands[frame_idx, :2])
        rot = np.rad2deg(commands[frame_idx, 2])
        moving_dist += dist

        # only update while the robot moves
        if dist > 0.01 or abs(rot) > 1:
            # recoding the estimated pose and ground truth pose only
            # when localization is converged and robot not staying (for evaluation)
            valid_frames.append(frame_idx)

            # ENM-based sensor model
            particles = sensor_model.update_weights(particles, frame_idx, T_b2l=T_b2l)

            # check convergence
            std = np.linalg.norm(np.std(particles[:, :2], axis=0))
            if std < 0.3 and not is_converged:
                is_converged = True
                offset = frame_idx
                converge_time_threshold = timestamps_gt[frame_idx] - timestamps_gt[start_idx]
                print('Initialization is finished after {:.2f}s'.format(converge_time_threshold))

                # cutoff redundant particles and leave only num of particles
                idxes = np.argsort(particles[:, 3])[::-1]
                particles = particles[idxes[:reduced_num]]

                # normalize the weights
                particles[:, 3] /= np.sum(particles[:, 3])

                # adaptive the sensor model's parameters
                sensor_model.lambda_avg = cfg_loc['lambda_adaptive']

            if visualize:
                visualizer.update(frame_idx, particles)
                visualizer.fig.canvas.draw()
                visualizer.fig.canvas.flush_events()

            # resampling
            particles = srrg_utils.resample(particles)

        else:
            if visualize:
                visualizer.update(frame_idx, particles)
                visualizer.fig.canvas.draw()
                visualizer.fig.canvas.flush_events()

        curr_numParticles = particles.shape[0]
        results[frame_idx, :curr_numParticles] = particles

        cost_time = np.round(time.time() - start, 10)

        if frame_idx in valid_frames:
            if not is_converged:
                pre_converged_times.append(cost_time)
            else:
                post_converged_times.append(cost_time)

        total_times.append(cost_time)

        progress_bar.set_postfix({
            'timestamp': f'{timestamps_gt[frame_idx]:.2f}s',
            'time cost': f'{cost_time:.4f}s'
        })

    # print the average processing time
    avg_pre_converged_time = np.mean(pre_converged_times) if pre_converged_times else 0
    avg_post_converged_time = np.mean(post_converged_times) if post_converged_times else 0
    avg_total_time = np.mean(total_times) if total_times else 0
    avg_pre_converged_freq = 1 / avg_pre_converged_time if avg_pre_converged_time > 0 else 0
    avg_post_converged_freq = 1 / avg_post_converged_time if avg_post_converged_time > 0 else 0
    avg_total_freq = 1 / avg_total_time if avg_total_time > 0 else 0
    print(f'Average processing frequency before convergence: {avg_pre_converged_freq:.2f} Hz')
    print(f'Average processing frequency after convergence: {avg_post_converged_freq:.2f} Hz')
    print(f'Average processing frequency for the whole process: {avg_total_freq:.2f} Hz')

    # keep the valid frames for evaluation
    timestamps_est = timestamps_gt[valid_frames]
    results = results[valid_frames]

    # evaluate localization results (through evo)
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))

    result_dir = os.path.dirname(result_path)
    summary_loc(results, start_idx, reduced_num, timestamps_est, timestamps_gt, result_dir,
                cfg_loc['gt_file'], cfg_map['map_type'], init_time_thres=20,
                use_converge=False, suffix=args.suffix)

    # save the valid GT frames for fair evaluation
    valid_frames_path = os.path.join(result_dir, 'GT.txt')
    gt_poses_tum = np.loadtxt(cfg_loc['gt_file'])
    np.savetxt(valid_frames_path, gt_poses_tum)

    print('save the localization results at: {}\n'.format(result_path))

    if cfg_loc['plot_loc_results']:
        plot_traj_result(results, poses_gt, grid_res=grid_res, occ_map=occmap,
                         numParticles=numParticles, start_idx=start_idx + offset)

    print("Finished!\n")
