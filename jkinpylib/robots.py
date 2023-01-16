from typing import List

from jkinpylib.robot import Robot
from jkinpylib.utils import get_filepath


# TODO(@jstmn): Fix batch FK for baxter
class Baxter(Robot):
    name = "baxter"
    formal_robot_name = "Baxter"

    def __init__(self):
        joint_chain = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2", "left_hand"]
        end_effector_link_name = "left_hand"
        urdf_filepath = get_filepath("urdfs/baxter/baxter.urdf")
        Robot.__init__(self, Baxter.name, urdf_filepath, joint_chain, end_effector_link_name, batch_fk_enabled=False)


class Fetch(Robot):
    name = "fetch"
    formal_robot_name = "Fetch"

    def __init__(self):
        # Sum joint range: 34.0079 rads
        joint_chain = [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",  # continuous
            "elbow_flex_joint",
            "forearm_roll_joint",  # continuous
            "wrist_flex_joint",
            "wrist_roll_joint",  # continous
            "gripper_axis",  # fixed
        ]
        end_effector_link_name = "gripper_link"
        urdf_filepath = get_filepath("urdfs/fetch/fetch_formatted.urdf")
        Robot.__init__(self, Fetch.name, urdf_filepath, joint_chain, end_effector_link_name)


# TODO: Add base link to 'Robot'
# class FetchNoPrismatic(Robot):
#     name = "fetch_no_prismatic"
#     formal_robot_name = "Fetch - No lift joint"

#     def __init__(self):
#         # Sum joint range:
#         joint_chain = [
#             # "torso_lift_joint",
#             "shoulder_pan_joint",
#             "shoulder_lift_joint",
#             "upperarm_roll_joint",  # continuous
#             "elbow_flex_joint",
#             "forearm_roll_joint",  # continuous
#             "wrist_flex_joint",
#             "wrist_roll_joint",  # continous
#             "gripper_axis",  # fixed
#         ]
#         end_effector_link_name = "gripper_link"
#         urdf_filepath = get_filepath("urdfs/fetch/fetch_formatted.urdf")
#         Robot.__init__(self, FetchNoPrismatic.name, urdf_filepath, joint_chain, end_effector_link_name)


class Panda(Robot):
    name = "panda"
    formal_robot_name = "Panda"

    def __init__(self, verbose: bool = False):
        joint_chain = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_joint8",
            "panda_hand_joint",
        ]
        urdf_filepath = get_filepath("urdfs/panda/panda_arm_hand_formatted.urdf")
        end_effector_link_name = "panda_hand"
        Robot.__init__(self, Panda.name, urdf_filepath, joint_chain, end_effector_link_name, verbose=verbose)


class Iiwa7(Robot):
    name = "iiwa7"
    formal_robot_name = "Kuka LBR IIWA7"

    def __init__(self, verbose: bool = False):
        joint_chain = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
            "iiwa_joint_ee",  # fixed
        ]
        urdf_filepath = get_filepath("urdfs/iiwa7/iiwa7_formatted.urdf")
        end_effector_link_name = "iiwa_link_ee"
        Robot.__init__(self, Iiwa7.name, urdf_filepath, joint_chain, end_effector_link_name, verbose=verbose)


ALL_CLCS = [Panda, Fetch, Iiwa7]
# ALL_CLCS = [Panda, Fetch, Baxter, FetchNoPrismatic]


def get_all_robots() -> List[Robot]:
    return [clc() for clc in ALL_CLCS]


def get_robot(robot_name: str) -> Robot:
    for clc in ALL_CLCS:
        if clc.name == robot_name:
            return clc()
    raise ValueError(f"Unable to find robot '{robot_name}' (available: {[clc.name for clc in ALL_CLCS]})")


def robot_name_to_fancy_robot_name(name: str) -> str:
    for cls in ALL_CLCS:
        if cls.name == name:
            return cls.formal_robot_name
    raise ValueError(f"Unable to find robot '{name}' (available: {[clc.name for clc in ALL_CLCS]})")


if __name__ == "__main__":
    r = Iiwa7(verbose=True)
