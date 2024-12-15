import argparse
from time import sleep

import numpy as np
from klampt import vis

from visualization_utils import _init_klampt_vis
from jrl.robot import Robot
from jrl.robots import Panda, Fetch, Rizon4, Ur5, get_robot, Ur3, XArm6


_TARGET_POSES = {
    Panda.name: np.array([0.25, 0.65, 0.45, 1.0, 0.0, 0.0, 0.0]),
    Fetch.name: np.array([0.45, 0.65, 0.55, 1.0, 0.0, 0.0, 0.0]),
    Rizon4.name: np.array([0.4, 0.4, 0.45, 1.0, 0.0, 0.0, 0.0]),
    Ur5.name: np.array([0.25, 0.65, 0.45, 1.0, 0.0, 0.0, 0.0]),
    Ur3.name: np.array([ 0.3, -0.4, 0.07, 0.00, 0.00, 0.00, 1.00]),
    XArm6.name: np.array([ 0.3, -0.4, 0.07, 0.00, 0.00, 1.00, 0.00]),
}


def show_joints(robot: Robot, joints: np.ndarray):
    """Fixed end pose with n=n_qs different solutions"""
    _init_klampt_vis(robot, f"{robot.formal_robot_name} - IK redundancy")
    vis.add("world", robot.klampt_world_model)
    # vis.setColor(vis.getItemName(robot.klampt_world_model.robot(0)), 0.7, 0.7, 0.7, 0.1)
    vis.hide(vis.getItemName(robot.klampt_world_model.robot(0)))

    qs = robot._x_to_qs(joints)
    for i, q in enumerate(qs):
        vis.add(f"robot_{i}", q)
        vis.setColor(f"robot_{i}", 0.7, 0.7, 0.7, 0.3)

    while vis.shown():
        vis.lock()
        vis.unlock()
        sleep(1 / 30)  # note: don't put sleep inside the lock()
    vis.kill()


""" Example usage:

python scripts/visualize_joints.py --robot_name=ur5
python scripts/visualize_joints.py --robot_name=panda
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--robot_name", type=str)
    args = parser.parse_args()
    robot = get_robot(args.robot_name)

    joints_all = np.array([[-1.4000, -0.4668, -0.4733, -0.9854, -0.2972,  0.6479,  0.7065],
            [-0.9808, -1.0183, -0.6218, -1.9987,  2.2476,  1.1838, -0.5449],
            [-1.1354,  0.9494, -2.5724, -1.3672,  2.7392,  3.7380,  0.8694],
            [ 0.6373, -0.4893, -2.0329, -0.1423,  2.3983,  0.0350, -2.7476],
            [-2.5984, -0.2952,  0.1339, -1.5887, -2.8419,  0.1984,  0.9625],
            [-1.6178, -0.1054, -0.8429, -0.5502,  0.2499,  0.2823,  0.1045],
            [-1.6397,  1.2193, -2.5536, -1.8490, -1.9930,  1.1175,  0.9051],
            [-0.2594, -0.8748, -0.8016, -1.2676,  1.5083,  0.3924, -0.3141],
            [ 2.2032, -0.6712,  0.2869, -1.1908,  0.9689,  0.6630,  0.1636],
            [ 0.1029, -1.4176, -0.0084, -1.7396, -1.3778,  1.4672,  1.8157]])
    joints_ur3 = np.array([[-0.70458942+np.pi,  3.5924112 , -0.96987947, -1.05173541,  1.57079633, -2.27538575]])
    joints_xarm6 = np.array([[-0.9254632489674508, 0.6990770671568564, -1.106629064060494, 0.0006653351931553931, 0.3987969742311386, -4.063402065624296,]])
    name = 'all' if args.robot_name in [Panda.name, Fetch.name, Rizon4.name, Ur5.name] else args.robot_name
    joints = {
                'all': joints_all,
                Ur3.name: joints_ur3,
                XArm6.name: joints_xarm6
            }
    show_joints(robot, joints[name][0:1])
