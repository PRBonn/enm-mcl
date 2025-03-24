"""The functions for supporting the MCL experiments
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved
"""

import json
import os.path

import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

from evo.core import metrics, sync
from evo.tools import file_interface


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


def load_data(pose_path, max_beams=None):
    # Read and parse the poses
    timestamps = []
    poses_gt = []
    odoms = []
    scans = []
    params = {}

    try:
        with open(pose_path, 'r') as f:
            all_data = json.load(f)

        # downsample beams number according to the max_beams
        if max_beams is not None:
            downsample_factor = all_data['num_beams'] // max_beams
            all_data['angle_res'] *= downsample_factor
            all_data['num_beams'] = max_beams
            all_data['angle_max'] = all_data['angle_min'] + all_data['angle_res'] * max_beams

        params.update({'num_beams': all_data['num_beams']})
        params.update({'angle_min': all_data['angle_min']})
        params.update({'angle_max': all_data['angle_max']})
        params.update({'angle_res': all_data['angle_res']})
        params.update({'max_range': all_data['max_range']})

        near = 0.02
        far = np.floor(all_data['max_range'])
        bound = np.array([near, far])
        # ray directions for all beams in the lidar coordinate, shape: (N, 2)
        directions = get_ray_directions(all_data['angle_min'], all_data['angle_max'],
                                        all_data['angle_res'])

        params.update({'near': near})
        params.update({'far': far})
        params.update({'bound': bound})
        params.update({'directions': directions})

        for data in all_data['scans']:
            timestamps.append(data['timestamp'])

            pose = data['pose_gt']
            poses_gt.append(pose)

            odom = data['odom_reading']
            odoms.append(odom)

            scan = np.array(data['range_reading'])
            if max_beams is not None:
                scan = scan[::downsample_factor][:max_beams]
            scan[scan >= all_data['max_range']] = 0
            scans.append(scan)

    except FileNotFoundError:
        print('Ground truth poses are not available.')

    return np.array(timestamps), np.array(poses_gt), np.array(odoms), np.array(scans), params


def particles2pose(particles):
    """
    Convert particles to the estimated pose accodring to the particles' distribution
    :param particles: 2-D array, (N, 4) shape
    :return: a estimated 2D pose, 1-D array, (3,) shape
    """
    normalized_weight = particles[:, 3] / np.sum(particles[:, 3])

    # average angle (https://vicrucann.github.io/tutorials/phase-average/)
    particles_mat = np.zeros_like(particles)
    particles_mat[:, :2] = particles[:, :2]
    particles_mat[:, 2] = np.cos(particles[:, 2])
    particles_mat[:, 3] = np.sin(particles[:, 2])
    estimated_pose_temp = particles_mat.T.dot(normalized_weight.T)

    estimated_pose = np.zeros(shape=(3,))
    estimated_pose[:2] = estimated_pose_temp[:2]
    estimated_pose[2] = np.arctan2(estimated_pose_temp[-1], estimated_pose_temp[-2])

    return estimated_pose


def get_est_poses(all_particles, start_idx, numParticles):
    estimated_traj = []
    ratio = 0.8

    for frame_idx in range(start_idx, all_particles.shape[0]):
        particles = all_particles[frame_idx]
        # collect top 80% of particles to estimate pose
        idxes = np.argsort(particles[:, 3])[::-1]
        idxes = idxes[:int(ratio * numParticles)]

        partial_particles = particles[idxes]
        if np.sum(partial_particles[:, 3]) == 0:
            continue

        estimated_pose = particles2pose(partial_particles)
        estimated_traj.append(estimated_pose)

    estimated_traj = np.array(estimated_traj)

    return estimated_traj


def convert2tum(timestamps, poses):
    tum_poses = []

    for t, pose in zip(timestamps, poses):
        x, y, yaw = pose
        q = R.from_euler('z', yaw).as_quat()
        curr_data = [t,
                     x, y, 0,
                     q[0], q[1], q[2], q[3]]

        tum_poses.append(curr_data)

    tum_poses = np.array(tum_poses)

    return tum_poses


def evaluate_APE(est_poses, gt_poses, use_converge=False):
    # align est and gt
    max_diff = 0.01
    traj_ref, traj_est = sync.associate_trajectories(gt_poses, est_poses, max_diff)
    data = (traj_ref, traj_est)

    # location error
    ape_location = metrics.APE(metrics.PoseRelation.translation_part)
    ape_location.process_data(data)
    location_errors = ape_location.error

    location_ptc5 = location_errors < 0.05
    location_ptc5 = np.sum(location_ptc5) / location_ptc5.shape[0] * 100

    location_ptc10 = location_errors < 0.1
    location_ptc10 = np.sum(location_ptc10) / location_ptc10.shape[0] * 100

    location_ptc20 = location_errors < 0.2
    location_ptc20 = np.sum(location_ptc20) / location_ptc20.shape[0] * 100

    location_rmse = ape_location.get_statistic(metrics.StatisticsType.rmse) * 100

    # yaw error
    ape_yaw = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_yaw.process_data(data)

    yaw_errors = ape_yaw.error
    yaw_ptc5 = yaw_errors < 0.5
    yaw_ptc5 = np.sum(yaw_ptc5) / yaw_ptc5.shape[0] * 100

    yaw_ptc10 = yaw_errors < 1.0
    yaw_ptc10 = np.sum(yaw_ptc10) / yaw_ptc10.shape[0] * 100

    yaw_ptc20 = yaw_errors < 2.0
    yaw_ptc20 = np.sum(yaw_ptc20) / yaw_ptc20.shape[0] * 100

    yaw_rmse = ape_yaw.get_statistic(metrics.StatisticsType.rmse)

    if use_converge:
        converge_idx = 0
        for idx in range(location_errors.shape[0]):
            if location_errors[idx] < 0.5 and yaw_errors[idx] < 10:
                converge_idx = idx
                break
        location_rmse = np.sqrt(np.mean(location_errors[converge_idx:] ** 2)) * 100
        yaw_rmse = np.sqrt(np.mean(yaw_errors[converge_idx:] ** 2))

    return location_rmse, location_ptc5, location_ptc10, location_ptc20, \
        yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20


def summary_loc(loc_results, start_idx, numParticles, timestamps_est, timestamps_gt, result_dir, gt_file,
                sensor_model='ENM', init_time_thres=5, use_converge=False, suffix=''):
    # convert loc_results to tum format
    timestamps_gt = timestamps_gt[start_idx:]
    init_time_thres += timestamps_gt[0]

    # get estimated poses
    est_poses = get_est_poses(loc_results, start_idx, numParticles)
    est_tum = convert2tum(timestamps_est, est_poses)

    # save est_traj in tum format
    est_tum_file = os.path.join(result_dir, '{}{}.txt'.format(sensor_model, suffix))
    np.savetxt(est_tum_file, est_tum)

    # evo evaluation
    print('\nEvaluation')

    # Estimated poses
    est_poses = file_interface.read_tum_trajectory_file(est_tum_file)
    est_poses.reduce_to_time_range(init_time_thres)

    # GT
    gt_poses = file_interface.read_tum_trajectory_file(gt_file)
    gt_poses.reduce_to_time_range(init_time_thres)

    print("Sequence information: ", est_poses)
    print(("{:>15}\t" * 8).format(
        "location_rmse", "location_ptc5", "location_ptc10", "location_ptc20",
        "yaw_rmse", "yaw_ptc5", "yaw_ptc10", "yaw_ptc20"))

    location_rmse, location_ptc5, location_ptc10, location_ptc20, \
        yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20 = \
        evaluate_APE(est_poses, gt_poses, use_converge=use_converge)

    # print error info
    print(("{:15.2f}\t" * 8).format(
        location_rmse, location_ptc5, location_ptc10, location_ptc20,
        yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20))


def particles2pose_with_covariance(particles):
    """
    Compute estimated pose and covariance matrix from particles.

    :param particles: np.array of shape (N, 4) -> [x, y, theta, weight]
    :return: estimated pose (3,), covariance matrix (36,)
    """
    weights = particles[:, 3]
    weights /= np.sum(weights)  # Normalize weights

    # Compute weighted mean position
    mean_x = np.sum(weights * particles[:, 0])
    mean_y = np.sum(weights * particles[:, 1])

    # Compute weighted mean of cos(theta) and sin(theta)
    mean_cos = np.sum(weights * np.cos(particles[:, 2]))
    mean_sin = np.sum(weights * np.sin(particles[:, 2]))
    mean_theta = np.arctan2(mean_sin, mean_cos)

    estimated_pose = np.array([mean_x, mean_y, mean_theta])

    # Compute covariance matrix
    centered_particles = particles[:, :2] - estimated_pose[:2]
    cov_xy = np.cov(centered_particles.T, aweights=weights)  # Compute 2x2 covariance

    # Compute Fisher Circular Variance for theta
    R = np.linalg.norm([mean_cos, mean_sin]) / np.sum(weights)
    cov_theta = 1 - min(1.0, R)  # Ensuring numerical stability

    # Expand to full 6x6 covariance matrix
    full_covariance = np.zeros((6, 6))
    full_covariance[:2, :2] = cov_xy  # (x, y)
    full_covariance[5, 5] = cov_theta  # (yaw)

    # Convert to 1D (flatten)
    covariance_flat = full_covariance.flatten()

    return estimated_pose, covariance_flat


def get_est_poses_with_covariance(all_particles, start_idx, numParticles):
    estimated_traj = []
    estimated_covs = []
    ratio = 0.8

    for frame_idx in range(start_idx, all_particles.shape[0]):
        particles = all_particles[frame_idx]

        # Select top 80% of particles based on weight
        idxes = np.argsort(particles[:, 3])[::-1]
        idxes = idxes[:int(ratio * numParticles)]
        partial_particles = particles[idxes]

        if np.sum(partial_particles[:, 3]) == 0:
            continue

        estimated_pose, covariance = particles2pose_with_covariance(partial_particles)
        estimated_traj.append(estimated_pose)
        estimated_covs.append(covariance)

    estimated_traj = np.array(estimated_traj)
    estimated_covs = np.array(estimated_covs)

    return estimated_traj, estimated_covs


def summary_loc_with_cov(loc_results, start_idx, numParticles, timestamps_est, result_dir,
                         sensor_model='ENM-MCL', suffix=''):
    # Convert localization results to TUM format with covariance

    # Get estimated poses and covariance
    est_poses, est_covs = get_est_poses_with_covariance(loc_results, start_idx, numParticles)

    # Convert to TUM format
    est_tum = []
    for i, (pose, cov) in enumerate(zip(est_poses, est_covs)):
        timestamp = timestamps_est[i]
        x, y, theta = pose

        # Convert yaw to quaternion (assuming z=0)
        qx, qy, qz, qw = 0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2)

        # Format: timestamp, x, y, z=0, qx, qy, qz, qw, 36 covariance values
        tum_line = [timestamp, x, y, 0.0, qx, qy, qz, qw] + cov.tolist()
        est_tum.append(tum_line)

    # Save in TUM format
    est_tum_file = os.path.join(result_dir, '{}{}_cov.txt'.format(sensor_model, suffix))
    np.savetxt(est_tum_file, est_tum)

    print(f"Saved trajectory with covariance at: {est_tum_file}")
