from time import sleep

from klampt import vis
from klampt.model import coordinates, trajectory
from klampt.math import so3
import torch
import numpy as np

from jkinpylib.robot import Robot
from jkinpylib.robots import Panda
from jkinpylib.utils import set_seed
from jkinpylib.config import device

set_seed()

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True)
_TERRAIN_FILEPATH = "scripts/visualization_resources/plane.off"


def _init_vis(robot: Robot, window_title: str):
    vis.init()
    assert robot.klampt_world_model.loadTerrain(_TERRAIN_FILEPATH), f"Failed to load terrain '{_TERRAIN_FILEPATH}'"
    vis.add("world", robot.klampt_world_model)
    vis.add("coordinates", coordinates.manager())
    vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
    vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))
    vis.setWindowTitle(window_title)
    vis.show()


def plot_pose(vis_name: str, pose: np.ndarray):
    vis.add(vis_name, (so3.from_quaternion(pose[3:]), pose[0:3]), length=0.15, width=2)


# TODO: Delete plot_batch_ik_optimization_step.py
""" 
python scripts/plot_batch_ik_optimization.py
"""

if __name__ == "__main__":
    robot = Panda()
    alpha = 0.1

    target_poses = robot.forward_kinematics(np.array([[1.0, 1.5707, 0, 0, 0, 3.141592, 0]]))
    # -> np.ndarray([[ 0.43601941,  0.67906,     0.42107767,  0.07573132, -0.4435556,  -0.55064561, -0.70307368]])

    # Get the current poses (these will be the seeds)
    x_current = np.array([[0, 1.5707, 0, 0, 0, 3.141592, 0]])
    _init_vis(robot, "IK Step")

    print("sleeping")
    # sleep(15)
    print("starting")

    plot_pose("target_pose", target_poses[0])
    plot_pose("inital_pose", robot.forward_kinematics(x_current)[0])

    i = 0
    while vis.shown():
        vis.lock()

        # Get the target poses
        print(f"Starting optimization step #{i}")
        current_poses = robot.forward_kinematics(x_current)
        l2_errs_original = np.linalg.norm(target_poses[:, 0:3] - current_poses[:, 0:3], axis=1)

        robot.set_klampt_robot_config(x_current[0])
        plot_pose("current_pose", robot.forward_kinematics(x_current)[0])
        print("current poses:\n", current_poses)

        # pytorch
        x_updated, _ = robot.inverse_kinematics_single_step_batch_pt(
            torch.tensor(target_poses, device=device, dtype=torch.float32),
            torch.tensor(x_current, device=device, dtype=torch.float32),
            alpha,
        )
        x_updated = x_updated.cpu().numpy()
        # numpy
        # x_updated, _ = robot.inverse_kinematics_single_step_batch_np(target_poses, x_current, alpha)

        updated_poses = robot.forward_kinematics(x_updated)
        l2_errs_final = np.linalg.norm(target_poses[:, 0:3] - updated_poses[:, 0:3], axis=1)
        l2_errs_diff = l2_errs_final - l2_errs_original

        print("updated_poses:\n", updated_poses)
        print("pose_difference:\n", updated_poses - current_poses)
        print("l2 errors initial:   ", l2_errs_original)
        print("l2 errors final:     ", l2_errs_final)
        print("l2 errors difference:", l2_errs_diff)

        x_current = x_updated
        vis.unlock()
        sleep(1)
        i += 1
    vis.kill()
