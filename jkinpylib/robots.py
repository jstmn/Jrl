from jkinpylib.kinematics import KinematicChain
from jkinpylib.utils import get_filepath


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
