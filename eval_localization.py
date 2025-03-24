"""This script evaluates the localization results of MCL.
@author: Haofei Kuang    [haofei.kuang@igg.uni-bonn.de]
Copyright (c) 2025 Haofei Kuang, all rights reserved
"""
import argparse
import glob
import os

import numpy as np

from evo.tools import file_interface

from localization.utils import evaluate_APE


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_dir', type=str, default='enm-mcl/results/ipblab/final/',
                        help='the path for the result files.')
    return parser.parse_args()


def trajectory_error(est_poses, gt_poses, method, use_converge=False):
    # evo evaluation
    location_rmse, location_ptc5, location_ptc10, location_ptc20, \
        yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20 = \
        evaluate_APE(est_poses, gt_poses, use_converge=use_converge)

    print(("{:<15}").format(method) + ("\t{:15.2f}" * 8).format(
        location_rmse, location_ptc5, location_ptc10, location_ptc20,
        yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20))

    return np.array([
        location_rmse, location_ptc5, location_ptc10, location_ptc20,
        yaw_rmse, yaw_ptc5, yaw_ptc10, yaw_ptc20
    ])


def eval(cfg):
    # load predicted trajectories
    result_dir = args.results_dir

    # Open a file to save the average results
    avg_results_file = os.path.join(result_dir, 'average_results.txt')
    # clear the file
    with open(avg_results_file, 'w') as f:
        f.write("")

    # Get all sequence directories
    sequence_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    sequence_dirs = sorted(sequence_dirs)

    # Iterate over each sequence directory
    for id, sequence_dir in enumerate(sequence_dirs):
        sequence_dir = os.path.join(result_dir, sequence_dir)
        # loading all est trajectory files (method + number format)
        est_trajec_files = sorted(glob.glob(os.path.join(sequence_dir, '*.txt')))

        # load ground truth trajectories
        gt_tum_file = os.path.join(sequence_dir, 'GT.txt')
        # start_time = np.loadtxt(gt_tum_file).item((0, 0))
        start_time = 20.0

        # GT
        gt_poses = file_interface.read_tum_trajectory_file(gt_tum_file)
        gt_poses.reduce_to_time_range(start_time)

        print("\nSequence information (GT): ", gt_poses)
        print("\t"*2 + ("{:>15}\t" * 8).format(
            "location_rmse", "location_ptc5", "location_ptc10", "location_ptc20",
            "yaw_rmse", "yaw_ptc5", "yaw_ptc10", "yaw_ptc20"))

        # Dictionary to store cumulative results
        results_dict = {}

        # Estimated poses
        for est_tum_file in est_trajec_files:
            method_name = '.'.join(est_tum_file.split('/')[-1].split('.')[:-1])
            method_base = ''.join([i for i in method_name if not i.isdigit()])

            if method_base == 'GT' or method_base == 'average_results':
                continue

            # If the method is not in the results_dict, initialize it
            if method_base not in results_dict:
                results_dict[method_base] = []

            est_poses = file_interface.read_tum_trajectory_file(est_tum_file)
            est_poses.reduce_to_time_range(start_time)

            # Get error for this trial and append to the list for the method
            error = trajectory_error(est_poses, gt_poses, method_name)
            results_dict[method_base].append(error)

        # plot "===" to separate the average results from the individual trials
        print("=" * 143)

        with open(avg_results_file, 'a') as f:
            f.write(f"Sequence: {id+1}\n")
            # Calculate the average error for each method
            for method, errors in results_dict.items():
                # Filter out results where location_rmse (index 0) > 0.5 meters
                filtered_errors = [err for err in errors if err[0] <= 30]

                if len(filtered_errors) == 0:
                    print(f"{method}_avg: No valid results after filtering")
                    f.write(f"{method}_avg: No valid results after filtering\n")
                else:
                    avg_error = np.mean(filtered_errors, axis=0)
                    std_error = np.std(filtered_errors, axis=0)

                    # Calculate success rate
                    success_rate = len(filtered_errors) / len(errors) * 100

                    # Only print RMSE values (location_rmse and yaw_rmse)
                    location_rmse_avg = avg_error[0]
                    location_rmse_std = std_error[0]
                    yaw_rmse_avg = avg_error[4]
                    yaw_rmse_std = std_error[4]

                    rmse_output = f"{method}_avg  {location_rmse_avg:.2f} ± {location_rmse_std:.2f} / {yaw_rmse_avg:.2f} ± {yaw_rmse_std:.2f}"
                    success_output = f" Success rate: {success_rate:.2f}%"

                    print(rmse_output + success_output)

                    # Write to file
                    f.write(rmse_output + success_output + '\n')

            f.write("====================================\n")


if __name__ == '__main__':
    args = get_args()
    eval(args)
