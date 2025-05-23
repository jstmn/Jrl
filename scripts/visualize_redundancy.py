import argparse
from time import sleep

import numpy as np
from klampt import vis

from visualization_utils import _init_klampt_vis
from jrl.robot import Robot
from jrl.robots import Panda, Fetch, Rizon4, Ur5, get_robot


_TARGET_POSES = {
    Panda.name: np.array([0.25, 0.65, 0.45, 1.0, 0.0, 0.0, 0.0]),
    Fetch.name: np.array([0.45, 0.65, 0.55, 1.0, 0.0, 0.0, 0.0]),
    Rizon4.name: np.array([0.4, 0.4, 0.45, 1.0, 0.0, 0.0, 0.0]),
    Ur5.name: np.array([0.25, 0.65, 0.45, 1.0, 0.0, 0.0, 0.0]),
}


def show_redundancy(robot: Robot, n_qs=15):
    """Fixed end pose with n=n_qs different solutions"""
    target_pose = _TARGET_POSES[robot.name]
    _init_klampt_vis(robot, f"{robot.formal_robot_name} - IK redundancy")
    vis.add("world", robot.klampt_world_model)
    vis.setColor(vis.getItemName(robot.klampt_world_model.robot(0)), 0.7, 0.7, 0.7, 1.0)

    xs = []
    for i in range(50):
        if len(xs) == n_qs:
            break
        x_new = robot.inverse_kinematics_klampt(target_pose)
        fk_old = robot.forward_kinematics_klampt(x_new)
        x_new = np.remainder(x_new + np.pi, 2 * np.pi) - np.pi
        assert np.linalg.norm((robot.forward_kinematics_klampt(x_new)[0] - fk_old[0])[0:3]) < 1e-3
        is_duplicate = False
        for x in xs:
            if np.linalg.norm(x_new - x) < 0.01:
                is_duplicate = True
        if not is_duplicate:
            xs.append(x_new[0])
    print("found", len(xs), "unique solutions")

    qs = robot._x_to_qs(np.array(xs))
    for i, q in enumerate(qs):
        vis.add(f"robot_{i}", q)
        vis.setColor(f"robot_{i}", 0.7, 0.7, 0.7, 1.0)

    while vis.shown():
        vis.lock()
        vis.unlock()
        sleep(1 / 30)  # note: don't put sleep inside the lock()
    vis.kill()


""" Example usage:

uv run python scripts/visualize_redundancy.py --robot_name=ur5
uv run python scripts/visualize_redundancy.py --robot_name=panda
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--robot_name", type=str)
    args = parser.parse_args()
    robot = get_robot(args.robot_name)
    show_redundancy(robot)
