from jkinpylib.robot import Robot
from jkinpylib.utils import get_filepath


# TODO(@jstmn): Fix batch FK for baxter
class Baxter(Robot):
    name = "baxter"
    formal_robot_name = "Baxter"

    def __init__(self):
        joint_chain = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2", "left_hand"]
        end_effector_link_name = "left_hand"
        urdf_filepath = get_filepath(f"urdfs/baxter/baxter.urdf")
        Robot.__init__(self, Baxter.name, urdf_filepath, joint_chain, end_effector_link_name)


class PandaArm(Robot):
    name = "panda_arm"
    formal_robot_name = "Panda"

    def __init__(self):
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
        urdf_filepath = get_filepath(f"urdfs/panda_arm/panda_arm_hand_formatted.urdf")
        end_effector_link_name = "panda_hand"
        Robot.__init__(self, PandaArm.name, urdf_filepath, joint_chain, end_effector_link_name)


class PandaArmStanford(Robot):
    name = "panda_arm_stanford"
    formal_robot_name = "Panda"

    def __init__(self):
        # joint_chain may include non actuated joints.
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
        urdf_filepath = get_filepath(f"urdfs/panda_arm_stanford/panda_formatted.urdf")
        end_effector_link_name = "panda_hand"
        Robot.__init__(self, PandaArmStanford.name, urdf_filepath, joint_chain, end_effector_link_name)


# ALL_CLCS = [PandaArm]
ALL_CLCS = [PandaArmStanford, PandaArm]
# ALL_CLCS = [PandaArmStanford, Baxter]


def get_all_robots():
    return [clc() for clc in ALL_CLCS]


def get_robot(robot_name: str) -> Robot:
    for clc in ALL_CLCS:
        if clc.name == robot_name:
            return clc()
    raise ValueError(f"Unable to find robot '{robot_name}'")


def robot_name_to_fancy_robot_name(name: str) -> str:
    for cls in ALL_CLCS:
        if cls.name == name:
            return cls.formal_robot_name


# if __name__ == "__main__":
#     r = Baxter()
