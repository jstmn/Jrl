from time import sleep

from klampt import vis
from klampt.model import coordinates, trajectory
from klampt.math import so3
import numpy as np

from jrl.robot import Robot
from jrl.robots import Panda
from jrl.utils import set_seed

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


""" 

python scripts/plot_batch_ik_optimization_step.py

"""


if __name__ == "__main__":
    robot = Panda()
    alpha = 0.25

    # Get the current poses (these will be the seeds)
    x_current = np.array([[0, 1.5707, 0, 0, 0, 3.141592, 0]])
    current_poses = robot.forward_kinematics(x_current)

    # Get the target poses
    target_poses = robot.forward_kinematics(np.array([[1.0, 1.5707, 0, 0, 0, 3.141592, 0]]))
    # -> np.ndarray([[ 0.43601941,  0.67906,     0.42107767,  0.07573132, -0.4435556,  -0.55064561, -0.70307368]])
    l2_errs_original = np.linalg.norm(target_poses[:, 0:3] - current_poses[:, 0:3], axis=1)
    print("target_poses:", target_poses)

    print("\n  ------ <fn>\n")
    x_updated, _ = robot.inverse_kinematics_single_step_batch_np(target_poses, x_current, alpha)
    print("\n  ------ </fn>\n")
    updated_poses = robot.forward_kinematics(x_updated)
    l2_errs_final = np.linalg.norm(target_poses[:, 0:3] - updated_poses[:, 0:3], axis=1)

    # print("x_current:\n", x_current)
    # print("x_updated:\n", x_updated)

    # print("\n-----")
    print("target poses: \n", target_poses)
    print("current poses:\n", current_poses)
    print("updated_poses:\n", updated_poses)
    # print("\n-----")
    print("l2 errors initial:", l2_errs_original)
    print("l2 errors final:  ", l2_errs_final)

    _init_vis(robot, "IK Step")
    target_pose = target_poses[0]
    current_pose = current_poses[0]
    updated_pose = updated_poses[0]
    plot_pose("target_pose", target_pose)
    plot_pose("current_pose", current_pose)
    plot_pose("updated_pose", updated_pose)

    while vis.shown():
        vis.lock()
        print("Setting initial pose")
        robot.set_klampt_robot_config(x_current[0])
        vis.unlock()
        sleep(10)

        vis.lock()
        print("Setting updated pose")
        robot.set_klampt_robot_config(x_updated[0])
        vis.unlock()
        sleep(10)

    vis.kill()
