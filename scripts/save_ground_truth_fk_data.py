import argparse

import numpy as np
import torch

from jkinpylib.robots import get_robot
from jkinpylib.robot import forward_kinematics_kinpy
from jkinpylib.conversions import geodesic_distance_between_quaternions_np
from jkinpylib.utils import set_seed

set_seed()


""" Example usage:

python scripts/save_ground_truth_fk_data.py --robot=fetch

"""

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", required=True, type=str)
    args = parser.parse_args()

    #
    robot = get_robot(args.robot_name)
    joint_angles = robot.sample_joint_angles(10)
    poses_kinpy = forward_kinematics_kinpy(robot, joint_angles)
    poses_klampt = robot.forward_kinematics_klampt(joint_angles)
    poses_batchfk, _ = robot.forward_kinematics_batch(torch.tensor(joint_angles, dtype=torch.float32, device="cuda"))

    np.testing.assert_allclose(poses_kinpy[:, 0:3], poses_klampt[:, 0:3])
    np.testing.assert_allclose(poses_kinpy[:, 0:3], poses_batchfk[:, 0:3, 3].detach().cpu().numpy(), atol=5e-4)
    rotational_errros = geodesic_distance_between_quaternions_np(poses_kinpy[:, 3:7], poses_klampt[:, 3:7])
    assert max(rotational_errros) < 5e-4, f"Error, max(rotational_errros) > 5e-4 ({max(rotational_errros)} vs 0.0005)"

    np.save(f"data/ground_truth_fk_data/{robot.name}__joint_angles.npy", joint_angles)
    np.save(f"data/ground_truth_fk_data/{robot.name}__poses.npy", poses_klampt)
