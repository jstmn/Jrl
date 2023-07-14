import unittest

from jrl.robots import Fetch, FetchArm
from jrl.urdf_utils import Joint, get_kinematic_chain, get_lowest_common_ancestor_link


def _get_joint(name: str, parent: str, child: str, joint_type: str):
    return Joint(
        name=name,
        parent=parent,
        child=child,
        origin_rpy=(0, 0, 0),
        origin_xyz=(0, 0, 0),
        axis_xyz=(0, 0, 0),
        joint_type=joint_type,
        limits=(0, 0),
        velocity_limit=1.0,
    )


class UrdfUtilsTest(unittest.TestCase):
    def test_additional_link(self):
        """Test that the kinematic chain returned from the end-effectors kinematic chain to the additional_link for
        Fetch, FetchArm is as expected.
        """
        for fetch in [Fetch(), FetchArm()]:
            self.assertEqual(fetch.additional_link, "head_tilt_link")

            # Test lowest common ancestor to end-effector kinematic chain
            expected = "torso_lift_link"
            returned = get_lowest_common_ancestor_link(
                fetch._urdf_filepath,
                fetch._end_effector_kinematic_chain,
                fetch._active_joints,
                fetch._additional_link_name,
            )
            self.assertEqual(expected, returned)

            # Test chain
            expected = [
                "head_pan_joint",
                "head_tilt_joint",
            ]
            returned = [joint.name for joint in fetch._addl_link_kinematic_chain]
            self.assertEqual(expected, returned)

    def test_iiwa7(self):
        # world_iiwa_joint -> iiwa_joint_1 -> ... -> iiwa_joint_7 -> iiwa_joint_ee
        active_joints = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ]
        urdf_filepath = "jrl/urdfs/iiwa7/iiwa7_formatted.urdf"
        base_link = "world"
        end_effector_link_name = "iiwa_link_ee"
        joint_chain = get_kinematic_chain(urdf_filepath, active_joints, base_link, end_effector_link_name)
        self.assertEqual(len(joint_chain), 9)

    def test_get_joint_chain(self):
        urdf_filepath = "jrl/urdfs/panda/panda_arm_hand_formatted.urdf"

        # Test 1:
        # Link+joint chain:     panda_link1 -> panda_joint2 -> panda_link2 -> panda_joint3 -> panda_link3 -> panda_joint4 -> panda_link4 -> panda_joint5 -> panda_link5
        # Expected joint chain: panda_joint2 -> panda_joint3 -> panda_joint4 -> panda_joint5
        #                                       ^ active        ^ active
        active_joints = ["panda_joint3", "panda_joint4"]
        base_link_name = "panda_link1"
        end_effector_name = "panda_link5"

        expected = [
            _get_joint("panda_joint2", "panda_link1", "panda_link2", "fixed"),
            _get_joint("panda_joint3", "panda_link2", "panda_link3", "revolute"),
            _get_joint("panda_joint4", "panda_link3", "panda_link4", "revolute"),
            _get_joint("panda_joint5", "panda_link4", "panda_link5", "fixed"),
        ]

        returned = get_kinematic_chain(urdf_filepath, active_joints, base_link_name, end_effector_name)

        for expected_joint, returned_joint in zip(expected, returned):
            self.assertEqual(expected_joint.name, returned_joint.name)
            self.assertEqual(expected_joint.parent, returned_joint.parent)
            self.assertEqual(expected_joint.child, returned_joint.child)
            self.assertEqual(expected_joint.joint_type, returned_joint.joint_type)


if __name__ == "__main__":
    unittest.main()
