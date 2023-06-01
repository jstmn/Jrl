from typing import List

import torch
import numpy as np

from jkinpylib.robot import Robot
from jkinpylib.utils import get_filepath
from jkinpylib.config import DEFAULT_TORCH_DTYPE, DEVICE


def _load_capsule(path: str):
    data = np.loadtxt(get_filepath(path), delimiter=",")
    return torch.tensor(data, dtype=DEFAULT_TORCH_DTYPE, device=DEVICE)


# TODO(@jstmn): Fix batch FK for baxter
class Baxter(Robot):
    name = "baxter"
    formal_robot_name = "Baxter"

    # See
    # Rotational repeatability calculated in calculate_rotational_repeatability.py
    POSITIONAL_REPEATABILITY_MM = 0.1
    ROTATIONAL_REPEATABILITY_DEG = -1  # TODO

    def __init__(self):
        active_joints = [
            "left_s0",
            "left_s1",
            "left_e0",
            "left_e1",
            "left_w0",
            "left_w1",
            "left_w2",
        ]

        base_link = "base"
        end_effector_link_name = "left_hand"

        urdf_filepath = get_filepath("urdfs/baxter/baxter.urdf")

        # TODO: set 'ignored_collision_pairs'
        ignored_collision_pairs = []
        Robot.__init__(
            self,
            Baxter.name,
            urdf_filepath,
            active_joints,
            base_link,
            end_effector_link_name,
            ignored_collision_pairs,
            Baxter.POSITIONAL_REPEATABILITY_MM,
            Baxter.ROTATIONAL_REPEATABILITY_DEG,
            batch_fk_enabled=False,
        )


class Fetch(Robot):
    name = "fetch"
    formal_robot_name = "Fetch"

    # Rotational repeatability calculated in calculate_rotational_repeatability.py
    POSITIONAL_REPEATABILITY_MM = 0.1
    ROTATIONAL_REPEATABILITY_DEG = 0.08296040224661197

    def __init__(self):
        # Sum joint range: 34.0079 rads
        active_joints = [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",  # continuous
            "elbow_flex_joint",
            "forearm_roll_joint",  # continuous
            "wrist_flex_joint",
            "wrist_roll_joint",  # continous
        ]
        base_link = "base_link"
        end_effector_link_name = "gripper_link"
        urdf_filepath = get_filepath("urdfs/fetch/fetch_formatted.urdf")
        ignored_collision_pairs = [
            ("torso_lift_link", "torso_fixed_link"),
            ("r_gripper_finger_link", "l_gripper_finger_link"),
            ("bellows_link2", "base_link"),
            ("bellows_link2", "torso_fixed_link"),
        ]
        Robot.__init__(
            self,
            Fetch.name,
            urdf_filepath,
            active_joints,
            base_link,
            end_effector_link_name,
            ignored_collision_pairs,
            Fetch.POSITIONAL_REPEATABILITY_MM,
            Fetch.ROTATIONAL_REPEATABILITY_DEG,
        )


class FetchArm(Robot):
    name = "fetch_arm"
    formal_robot_name = "Fetch - Arm (no lift joint)"

    # See
    # Rotational repeatability calculated in calculate_rotational_repeatability.py
    POSITIONAL_REPEATABILITY_MM = 0.1
    ROTATIONAL_REPEATABILITY_DEG = 0.10705219156268285

    def __init__(self, verbose: bool = False):
        # Sum joint range: 33.6218 rads
        active_joints = [
            "shoulder_pan_joint",  # (-1.6056, 1.6056)
            "shoulder_lift_joint",  # (-1.221,  1.518)
            "upperarm_roll_joint",  # (-3.1415, 3.1415) continuous
            "elbow_flex_joint",  # (-2.251,  2.251)
            "forearm_roll_joint",  # (-3.1415, 3.1415) continuous
            "wrist_flex_joint",  # (-2.16,   2.16)
            "wrist_roll_joint",  # (-3.1415, 3.1415) continous
        ]
        base_link = "base_link"
        end_effector_link_name = "gripper_link"
        urdf_filepath = get_filepath("urdfs/fetch/fetch_formatted.urdf")
        ignored_collision_pairs = [
            ("torso_lift_link", "torso_fixed_link"),
            ("r_gripper_finger_link", "l_gripper_finger_link"),
            ("bellows_link2", "base_link"),
            ("bellows_link2", "torso_fixed_link"),
        ]
        Robot.__init__(
            self,
            FetchArm.name,
            urdf_filepath,
            active_joints,
            base_link,
            end_effector_link_name,
            ignored_collision_pairs,
            FetchArm.POSITIONAL_REPEATABILITY_MM,
            FetchArm.ROTATIONAL_REPEATABILITY_DEG,
            verbose=verbose,
        )


class Panda(Robot):
    name = "panda"
    formal_robot_name = "Panda"

    # See 'Pose repeatability' in https://pkj-robotics.dk/wp-content/uploads/2020/09/Franka-Emika_Brochure_EN_April20_PKJ.pdf.
    # Rotational repeatability calculated in calculate_rotational_repeatability.py
    POSITIONAL_REPEATABILITY_MM = 0.1
    ROTATIONAL_REPEATABILITY_DEG = 0.14076593566091963

    def __init__(self, verbose: bool = False):
        active_joints = [
            "panda_joint1",  # (-2.8973, 2.8973)
            "panda_joint2",  # (-1.7628, 1.7628)
            "panda_joint3",  # (-2.8973, 2.8973)
            "panda_joint4",  # (-3.0718, -0.0698)
            "panda_joint5",  # (-2.8973, 2.8973)
            "panda_joint6",  # (-0.0175, 3.7525)
            "panda_joint7",  # (-2.8973, 2.8973)
        ]

        # Must match the total number of joints (including fixed) in the robot.
        # Use "None" for no collision geometry
        collision_capsules_by_link = {
            "panda_link0": _load_capsule("urdfs/panda/capsules/link0.txt"),
            "panda_link1": _load_capsule("urdfs/panda/capsules/link1.txt"),
            "panda_link2": _load_capsule("urdfs/panda/capsules/link2.txt"),
            "panda_link3": _load_capsule("urdfs/panda/capsules/link3.txt"),
            "panda_link4": _load_capsule("urdfs/panda/capsules/link4.txt"),
            "panda_link5": _load_capsule("urdfs/panda/capsules/link5.txt"),
            "panda_link6": _load_capsule("urdfs/panda/capsules/link6.txt"),
            "panda_link7": _load_capsule("urdfs/panda/capsules/link7.txt"),
            # "panda_hand": _load_capsule("urdfs/panda/capsules/hand.txt"),
        }

        urdf_filepath = get_filepath("urdfs/panda/panda_arm_hand_formatted.urdf")
        base_link = "panda_link0"
        end_effector_link_name = "panda_hand"
        ignored_collision_pairs = [
            ("panda_hand", "panda_link7"),
            ("panda_rightfinger", "panda_leftfinger"),
            ("panda_link7", "panda_link5") # these two don't collide if joint limits are respected
        ]
        Robot.__init__(
            self,
            Panda.name,
            urdf_filepath,
            active_joints,
            base_link,
            end_effector_link_name,
            ignored_collision_pairs,
            Panda.POSITIONAL_REPEATABILITY_MM,
            Panda.ROTATIONAL_REPEATABILITY_DEG,
            collision_capsules_by_link,
            verbose=verbose,
        )


class Iiwa7(Robot):
    name = "iiwa7"
    formal_robot_name = "Kuka LBR IIWA7"

    # See
    # Rotational repeatability calculated in calculate_rotational_repeatability.py
    POSITIONAL_REPEATABILITY_MM = 0.1
    ROTATIONAL_REPEATABILITY_DEG = 0.12614500942996015

    def __init__(self, verbose: bool = False):
        active_joints = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ]
        urdf_filepath = get_filepath("urdfs/iiwa7/iiwa7_formatted.urdf")
        base_link = "world"
        end_effector_link_name = "iiwa_link_ee"

        ignored_collision_pairs = []
        Robot.__init__(
            self,
            Iiwa7.name,
            urdf_filepath,
            active_joints,
            base_link,
            end_effector_link_name,
            ignored_collision_pairs,
            Iiwa7.POSITIONAL_REPEATABILITY_MM,
            Iiwa7.ROTATIONAL_REPEATABILITY_DEG,
            verbose=verbose,
        )


ALL_CLCS = [Panda, Fetch, FetchArm, Iiwa7]
# ALL_CLCS = [Panda, Fetch, FetchArm, Iiwa7, Baxter]


def get_all_robots() -> List[Robot]:
    return [clc() for clc in ALL_CLCS]


def get_robot(robot_name: str) -> Robot:
    for clc in ALL_CLCS:
        if clc.name == robot_name:
            return clc()
    raise ValueError(
        f"Unable to find robot '{robot_name}' (available: {[clc.name for clc in ALL_CLCS]})"
    )


def robot_name_to_fancy_robot_name(name: str) -> str:
    for cls in ALL_CLCS:
        if cls.name == name:
            return cls.formal_robot_name
    raise ValueError(
        f"Unable to find robot '{name}' (available: {[clc.name for clc in ALL_CLCS]})"
    )


if __name__ == "__main__":
    import numpy as np

    np.set_printoptions(suppress=True)
    fa = FetchArm()
    ff = FetchArm()
    print(fa.forward_kinematics_klampt(np.zeros(7)[None, :], "torso_lift_link"))
    print(ff.forward_kinematics_klampt(np.zeros(7)[None, :], "torso_lift_link"))
