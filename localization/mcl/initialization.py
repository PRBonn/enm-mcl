"""The functions for particles initialization
author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved.
"""

import numpy as np
import torch

np.random.seed(0)


def generate_random_positions(map_size, num_points=100000):
    min_x, max_x, min_y, max_y = map_size
    # Generate random x and y coordinates
    x_coords = np.random.rand(num_points) * (max_x - min_x) + min_x
    y_coords = np.random.rand(num_points) * (max_y - min_y) + min_y

    # Combine the x and y coordinates to form positions
    positions = np.stack((x_coords, y_coords), axis=1)

    return positions


def sample_free_space_numpy(map_size, map_model, threshold_free=0.5, threshold_unknown=1.0,
                            num_points=100000):
    free_positions = []

    while len(free_positions) < num_points:
        # Sample a random position
        positions = generate_random_positions(
            map_size, num_points=(num_points - len(free_positions)))

        # Evaluate occupancy probability
        input_pos = torch.from_numpy(positions).float().to(map_model.device)
        sdf = map_model.predict(input_pos)
        mask = torch.logical_and(threshold_free < sdf, sdf < threshold_unknown)
        mask = mask.cpu().numpy()

        # Append free positions
        free_positions.extend(positions[mask].tolist())

    return np.array(free_positions[:num_points])


def init_particles_uniform(map_size, numParticles, map_representation=None,
                           threshold_free=0.5, threshold_unknown=1.0):
    """ Initialize particles uniformly.
      Args:
        map_size: size of the map.
        numParticles: number of particles.
      Return:
        particles.
    """
    if map_representation is None:
        positions = generate_random_positions(map_size, numParticles)
    else:
        positions = sample_free_space_numpy(
            map_size, map_representation, threshold_free, threshold_unknown, numParticles)
    yaws = -np.pi + 2 * np.pi * np.random.rand(numParticles, 1)
    weights = np.ones((numParticles, 1))
    particels = np.concatenate((positions, yaws, weights), axis=1)

    return particels


def init_particles_pose_tracking(numParticles, init_pose, noises=[2, 2, np.pi / 6.0], init_weight=1.0):
    """ Initialize particles with a noisy initial pose.
    Here, we use ground truth pose with noises defaulted as [±5 meters, ±5 meters, ±π/6 rad]
    to mimic a non-accurate GPS information as a coarse initial guess of the global pose.
    Args:
      numParticles: number of particles.
      init_pose: initial pose.
      noises: range of noises.
      init_weight: initialization weight.
    Return:
      particles.
    """
    mu = np.array(init_pose)
    cov = np.diag(noises)
    particles = np.random.multivariate_normal(mean=mu, cov=cov, size=numParticles)
    init_weights = np.ones((numParticles, 1)) * init_weight
    particles = np.hstack((particles, init_weights))

    return np.array(particles, dtype=float)


def remove_outliers(particles, map_model, threshold_free=0.5, threshold_unknown=1.0):
    """ Remove outliers from particles.
    Args:
      particles: particles, an N*4 array where each particle is [x, y, yaw, w].
      map_model: map model for predicting whether a particle is in a free space.
      threshold_free: threshold indicating free space.
      threshold_unknown: threshold indicating unknown space.
    Returns:
      particles: with invalid particles replaced by resampled valid particles.
    """
    # Extract positions and weights from particles
    positions = particles[:, :2]  # Only x, y for positions
    weights = particles[:, 3]  # Weight of each particle

    # Convert positions to tensor and predict the space status
    input_pos = torch.from_numpy(positions).float().to(map_model.device)
    sdf = map_model.predict(input_pos)

    # Mask particles in free space (valid) and in unknown or occupied space (invalid)
    mask = torch.logical_and(threshold_free < sdf, sdf < threshold_unknown)
    mask = mask.cpu().numpy()

    # Get valid particles and weights
    valid_particles = particles[mask]
    valid_weights = weights[mask]

    if valid_weights.size == 0:
        raise ValueError("No valid particles found.")

    # Normalize the weights of valid particles for resampling
    valid_weights = valid_weights / np.sum(valid_weights)

    # Number of invalid particles
    num_invalid_particles = np.sum(~mask)

    # Resample from valid particles to replace invalid particles
    resampled_indices = np.random.choice(len(valid_particles), size=num_invalid_particles, p=valid_weights)
    resampled_particles = valid_particles[resampled_indices]

    # Replace invalid particles with resampled particles
    particles[~mask] = resampled_particles

    return particles
