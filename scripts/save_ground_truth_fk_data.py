from jkinpylib.robots import get_all_robots
from jkinpylib.robot import forward_kinematics_kinpy
from jkinpylib.math_utils import geodesic_distance_between_quaternions
from jkinpylib.utils import set_seed

import numpy as np

set_seed()


""" Example usage:

python scripts/save_ground_truth_fk_data.py

"""

if __name__ == "__main__":
    for robot in get_all_robots():
        joint_angles = robot.sample_joint_angles(10)
        poses_kinpy = forward_kinematics_kinpy(robot, joint_angles)
        poses_klampt = robot.forward_kinematics_klampt(joint_angles)

        np.testing.assert_allclose(poses_kinpy[:, 0:3], poses_klampt[:, 0:3])
        rotational_errros = geodesic_distance_between_quaternions(poses_kinpy[:, 3:7], poses_klampt[:, 3:7])
        assert (
            max(rotational_errros) < 5e-4
        ), f"Error, max(rotational_errros) > 5e-4 ({max(rotational_errros)} vs 0.0005)"

        np.save(f"data/ground_truth_fk_data/{robot.name}__joint_angles.npy", joint_angles)
        np.save(f"data/ground_truth_fk_data/{robot.name}__poses.npy", poses_klampt)
