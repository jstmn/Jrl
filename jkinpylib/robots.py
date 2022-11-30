from jkinpylib.kinematics import KinematicChain
from jkinpylib.utils import get_filepath


# TODO(@jstmn): Fix batch FK for baxter
# class Baxter(KinematicChain):
#     name = "baxter"
#     formal_robot_name = "Baxter"

#     def __init__(self):
#         joint_chain = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2", "left_hand"]
#         end_effector_link_name = "left_hand"
#         urdf_filepath = get_filepath(f"urdfs/baxter/baxter.urdf")
#         KinematicChain.__init__(self, Baxter.name, urdf_filepath, joint_chain, end_effector_link_name)


class PandaArm(KinematicChain):
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
        urdf_filepath = get_filepath(f"urdfs/panda_arm/panda.urdf")
        end_effector_link_name = "panda_hand"
        KinematicChain.__init__(self, PandaArm.name, urdf_filepath, joint_chain, end_effector_link_name)


def get_all_robots():
    return [PandaArm()]
    # return [PandaArm(), Baxter()]


def get_robot(robot_name: str) -> KinematicChain:
    classes = [PandaArm]
    # classes = [PandaArm, Baxter]
    for clc in classes:
        if clc.name == robot_name:
            return clc()
    raise ValueError(f"Unable to find robot '{robot_name}'")
