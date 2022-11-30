import unittest

from jkinpylib.kinematics import KinematicChain
from jkinpylib.kinematics_utils import _len3_tuple_from_str
from jkinpylib.robots import get_all_robots

import torch

ROBOTS = get_all_robots()


class KinematicChainTest(unittest.TestCase):
    def test_list_from_str(self):
        inputs = [
            "0 0 0.333",
            "0 0 1.0",
            "0 0 1",
        ]
        expected_outputs = [(0, 0, 0.333), (0, 0, 1.0), (0, 0, 1.0)]
        for input_, expected_output in zip(inputs, expected_outputs):
            output = _len3_tuple_from_str(input_)
            self.assertIsInstance(output, tuple)
            self.assertEqual(len(output), 3)
            self.assertAlmostEqual(output[0], expected_output[0])
            self.assertAlmostEqual(output[1], expected_output[1])
            self.assertAlmostEqual(output[2], expected_output[2])

    def test_n_dofs(self):
        ground_truth_n_dofs = {
            "panda_arm": 7,
            "baxter": 7,
        }
        for robot in ROBOTS:
            self.assertEqual(robot.n_dofs, ground_truth_n_dofs[robot.name])

    def test_joint_limits(self):
        ground_truth_joint_limits = {
            "panda_arm": [
                (-2.9671, 2.9671),
                (-1.8326, 1.8326),
                (-2.9671, 2.9671),
                (-3.1416, 0.0873),
                # (-3.1416, 0.0), # Nov28 2022: NOTE: the upper limit for `panda_joint4` was previously mistakenly set to 0.0.
                (-2.9671, 2.9671),
                (-0.0873, 3.8223),
                (-2.9671, 2.9671),
            ],
            "baxter": [
                (-1.70167993878, 1.70167993878),
                (-2.147, 1.047),
                (-3.05417993878, 3.05417993878),
                (-0.05, 2.618),
                (-3.059, 3.059),
                (-1.57079632679, 2.094),
                (-3.059, 3.059),
            ],
        }
        for robot in ROBOTS:
            self.assertEqual(len(robot._joint_limits), robot.n_dofs)
            for gt_limit, parsed_limit in zip(ground_truth_joint_limits[robot.name], robot._joint_limits):
                self.assertAlmostEqual(gt_limit[0], parsed_limit[0])
                self.assertAlmostEqual(gt_limit[1], parsed_limit[1])

    def test_actuated_joint_names(self):
        ground_truth_actuated_joints = {
            "panda_arm": [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
            "baxter": [
                "left_s0",
                "left_s1",
                "left_e0",
                "left_e1",
                "left_w0",
                "left_w1",
                "left_w2",
            ],
        }
        for robot in ROBOTS:
            self.assertEqual(len(robot.actuated_joint_names), robot.n_dofs)
            self.assertListEqual(robot.actuated_joint_names, ground_truth_actuated_joints[robot.name])


if __name__ == "__main__":
    unittest.main()
