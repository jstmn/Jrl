from typing import List
import argparse
from time import sleep

from klampt import vis
from klampt.model import coordinates, trajectory
from klampt.math import so3
import numpy as np

from jkinpylib.robot import Robot
from jkinpylib.robots import get_robot

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


# TODO: Make this less creepy
def oscillate_joints(robot: Robot):
    """Move the robot around"""

    inc = 0.02
    time_p_loop = 1 / 60  # 60Hz, in theory
    klampt_robot = robot.klampt_world_model.robot(0)
    _init_vis(robot, "oscillate joints")

    x = np.array([(u + l) / 2.0 for (l, u) in robot.actuated_joints_limits])

    def update_robot(_x):
        vis.lock()
        robot.set_klampt_robot_config(_x)
        vis.unlock()
        sleep(time_p_loop)  # note: don't put sleep inside the lock()

    while vis.shown():
        for i in range(robot.n_dofs):
            l, u = robot.actuated_joints_limits[i]

            while x[i] < u:
                x[i] += inc
                update_robot(x)

            while x[i] > l:
                x[i] -= inc
                update_robot(x)

            while x[i] < (u + l) / 2.0:
                x[i] += inc
                update_robot(x)
    vis.kill()


def transition_between(robot: Robot, configs: List[List[float]]):
    """Move the robot around"""
    assert len(configs) >= 2

    time_p_loop = 0.005
    ratio_inc = 0.0025
    _init_vis(robot, "transition between target configs")

    target_xs = [np.array(config) for config in configs]
    target_xs_poses = robot.forward_kinematics(np.array(target_xs))

    for i, pose in enumerate(target_xs_poses):
        vis.add(f"ee_{i}", (so3.from_quaternion(pose[3:]), pose[0:3]), length=0.15, width=2)

    current_idx = 0

    def update_robot(_prev_x: np.ndarray, _next_x: np.ndarray, _ratio: float):
        x = _prev_x + _ratio * (_next_x - _prev_x)
        vis.lock()
        robot.set_klampt_robot_config(x)
        vis.unlock()
        sleep(time_p_loop)  # note: don't put sleep inside the lock()

    ratio = 0
    while vis.shown():
        prev_x = target_xs[current_idx % len(configs)]
        next_x = target_xs[(current_idx + 1) % len(configs)]
        update_robot(prev_x, next_x, ratio)

        ratio += ratio_inc
        if ratio >= 1.0:
            print(f"done for current_idx {current_idx}")
            current_idx += 1
            ratio = 0.0

    vis.kill()


""" Example usage

# Oscillate joints
python scripts/visualize_robot.py --robot_name=panda_arm
python scripts/visualize_robot.py --robot_name=panda_arm_stanford
python scripts/visualize_robot.py --robot_name=baxter

# Move between configs
python scripts/visualize_robot.py \
    --robot_name=panda_arm \
    --start_config 0   1.5707 0 0 0 3.141592 0 \
    --end_config   1.0 1.5707 0 0 0 3.141592 0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--robot_name", type=str, help="Example: 'panda_arm', 'baxter', ...")
    parser.add_argument(
        "--start_config",
        nargs="+",
        type=float,
        help=(
            "Start config of the robot. If provided the robot will move between this config and second config provided"
            " by '--end_config'"
        ),
    )
    parser.add_argument("--end_config", nargs="+", type=float, help="End config of the robot")
    args = parser.parse_args()
    assert (args.start_config is None and args.end_config is None) or (
        args.start_config is not None and args.end_config is not None
    ), "--start_config and --end_config must either have both provided, or neither be provided"

    robot = get_robot(args.robot_name)

    if args.start_config is None:
        oscillate_joints(robot)

    if args.start_config is not None:
        transition_between(robot, [args.start_config, args.end_config])
