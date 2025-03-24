import argparse

import numpy as np
import torch

from jrl.robots import get_robot
from jrl.robot import forward_kinematics_kinpy
from jrl.math_utils import geodesic_distance_between_quaternions
from jrl.utils import set_seed

set_seed()


""" Example usage:

uv run python scripts/save_ground_truth_fk_data.py --robot=fetch
uv run python scripts/save_ground_truth_fk_data.py --robot=fetch_arm
uv run python scripts/save_ground_truth_fk_data.py --robot=iiwa7
uv run python scripts/save_ground_truth_fk_data.py --robot=iiwa14
uv run python scripts/save_ground_truth_fk_data.py --robot=baxter
uv run python scripts/save_ground_truth_fk_data.py --robot=rizon4

"""

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", required=True, type=str)
    args = parser.parse_args()

    #
    robot = get_robot(args.robot_name)
    joint_angles = robot.sample_joint_angles(10)
    poses_kinpy = torch.tensor(forward_kinematics_kinpy(robot, joint_angles), dtype=torch.float32)
    poses_klampt = torch.tensor(robot.forward_kinematics_klampt(joint_angles), dtype=torch.float32)

    # Check that klampt, kinpy, and batchfk agree before saving
    torch.testing.assert_close(poses_kinpy[:, 0:3], poses_klampt[:, 0:3])
    rotational_errors_kinpy_klampt = geodesic_distance_between_quaternions(poses_kinpy[:, 3:7], poses_klampt[:, 3:7])
    assert max(rotational_errors_kinpy_klampt) < 0.001, (
        f"Error, max(rotational_errors_kinpy_klampt) > 0.001 ({max(rotational_errors_kinpy_klampt)})"
    )

    # batch_fk
    if robot._batch_fk_enabled:
        poses_batchfk = robot.forward_kinematics(
            torch.tensor(joint_angles, dtype=torch.float32, device="cuda"), return_quaternion=True
        )
        rotational_errors_kinpy_batchfk = geodesic_distance_between_quaternions(
            poses_kinpy[:, 3:7], poses_batchfk[:, 3:7]
        )
        assert max(rotational_errors_kinpy_batchfk) < 0.001, (
            f"Error, max(rotational_errors_kinpy_batchfk) > 0.001 ({max(rotational_errors_kinpy_batchfk)})"
        )
        torch.testing.assert_close(poses_kinpy[:, 0:3], poses_batchfk[:, 0:3], atol=5e-4, rtol=0.0)

    np.save(f"tests/ground_truth_fk_data/{robot.name}__joint_angles.npy", joint_angles)
    np.save(f"tests/ground_truth_fk_data/{robot.name}__poses.npy", poses_klampt.cpu().numpy())
