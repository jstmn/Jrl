from typing import List
import argparse
from time import sleep

from klampt import vis
from klampt.model import coordinates, trajectory
from klampt.math import so3
import numpy as np

from jrl.robot import Robot
from jrl.robots import get_robot


def _init_vis(robot: Robot, window_title: str, show_collision_capsules: bool = True):
    vis.init()

    background_color = (1, 1, 1, 0.7)
    vis.setBackgroundColor(background_color[0], background_color[1], background_color[2], background_color[3])
    size = 5
    for x0 in range(-size, size + 1):
        for y0 in range(-size, size + 1):
            vis.add(
                f"floor_{x0}_{y0}",
                trajectory.Trajectory([1, 0], [(-size, y0, 0), (size, y0, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )
            vis.add(
                f"floor_{x0}_{y0}2",
                trajectory.Trajectory([1, 0], [(x0, -size, 0), (x0, size, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )

    vis.add("world", robot.klampt_world_model)
    vis.add("coordinates", coordinates.manager())
    vis.setWindowTitle(window_title)
    vis.show()


# TODO: Make this less creepy
def oscillate_joints(robot: Robot, show_collision_capsules: bool = True):
    """Move the robot around"""

    inc = 0.0025
    time_p_loop = 1 / 60  # 60Hz, in theory

    _init_vis(robot, "oscillate joints", show_collision_capsules=show_collision_capsules)

    x = np.array([(u + l) / 2.0 for (l, u) in robot.actuated_joints_limits])

    def update_robot(_x):
        vis.lock()
        robot.set_klampt_robot_config(_x)
        vis.unlock()
        sleep(time_p_loop)  # note: don't put sleep inside the lock()

    while vis.shown():
        for i in range(robot.ndof):
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


def transition_between(robot: Robot, configs: List[List[float]], show_collision_capsules: bool = True):
    """Move the robot around"""
    assert len(configs) >= 2

    time_p_loop = 0.005
    ratio_inc = 0.0025
    _init_vis(robot, "transition between target configs", show_collision_capsules=show_collision_capsules)

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
python scripts/visualize_robot.py --robot_name=panda
python scripts/visualize_robot.py --robot_name=baxter
python scripts/visualize_robot.py --robot_name=iiwa7
python scripts/visualize_robot.py --robot_name=fetch
python scripts/visualize_robot.py --robot_name=rizon4
python scripts/visualize_robot.py --robot_name=ur5

# Move between configs
python scripts/visualize_robot.py \
    --robot_name=panda \
    --start_config 0   1.5707 0 0 0 3.141592 0 \
    --end_config   1.0 1.5707 0 0 0 3.141592 0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--robot_name", type=str, help="Example: 'panda', 'baxter', ...")
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
    parser.add_argument("--show_collision_capsules", default="false", type=str)
    args = parser.parse_args()
    args.show_collision_capsules = args.show_collision_capsules.upper() == "TRUE"
    assert (args.start_config is None and args.end_config is None) or (
        args.start_config is not None and args.end_config is not None
    ), "--start_config and --end_config must either have both provided, or neither be provided"
    assert (
        not args.show_collision_capsules
    ), "--show_collision_capsules currently unimplemented. bug @jstmn if you want this feature"

    robot = get_robot(args.robot_name)

    if args.start_config is None:
        oscillate_joints(robot, show_collision_capsules=args.show_collision_capsules)

    if args.start_config is not None:
        transition_between(
            robot, [args.start_config, args.end_config], show_collision_capsules=args.show_collision_capsules
        )
