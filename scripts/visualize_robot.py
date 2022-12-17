import argparse
from time import sleep

from jkinpylib.robot import Robot
from jkinpylib.robots import get_robot

from klampt import vis
import numpy as np


# TODO: Make this less creepy
def oscillate_joints(robot: Robot):
    """Move the robot around"""

    inc = 0.02
    time_p_loop = 1 / 60  # 60Hz, in theory
    klampt_robot = robot.klampt_world_model.robot(0)

    vis.init()
    vis.add("world", robot.klampt_world_model)
    # vis.add("robot", robot.klampt_world_model.robot(0))
    # vis.setColor("robot", 1, 0.1, 0.1, 1)
    vis.setWindowTitle(f"{robot.name} visualizer")
    vis.show()

    x = np.array([(u + l) / 2.0 for (l, u) in robot.actuated_joints_limits])

    def update_robot(_x):
        vis.lock()
        q = robot._x_to_qs(np.reshape(_x, (1, _x.size)))[0]
        klampt_robot.setConfig(q)
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


""" Example usage

python scripts/visualize_robot.py --robot_name=panda_arm
python scripts/visualize_robot.py --robot_name=panda_arm_stanford
python scripts/visualize_robot.py --robot_name=baxter
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="evaluate.py - evaluates IK models")
    parser.add_argument("--robot_name", type=str, help="Example: 'panda_arm', 'baxter', ...")
    args = parser.parse_args()

    robot = get_robot(args.robot_name)
    oscillate_joints(robot)
