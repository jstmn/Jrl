import unittest

from jkinpylib.kinematics import KinematicChain
from jkinpylib.kinematics_utils import _len3_tuple_from_str
from jkinpylib.robots import PandaArm

import torch


class KinematicChainTest(unittest.TestCase):
    def setUp(self):
        self.panda = PandaArm()

        self.robot_gt_joint_limits = {
            "panda_arm": [
                (-2.9671, 2.9671),
                (-1.8326, 1.8326),
                (-2.9671, 2.9671),
                (-3.1416, 0.0873),
                # (-3.1416, 0.0), # Nov28 2022: NOTE: the upper limit for `panda_joint4` was previously mistakenly set to 0.0.
                (-2.9671, 2.9671),
                (-0.0873, 3.8223),
                (-2.9671, 2.9671),
            ]
        }

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
        self.assertEqual(self.panda.n_dofs, 7)

    def test_joint_limits(self):
        robot = self.panda
        self.assertEqual(len(robot._joint_limits), robot.n_dofs)
        for gt_limit, parsed_limit in zip(self.robot_gt_joint_limits[robot.name], robot._joint_limits):
            self.assertAlmostEqual(gt_limit[0], parsed_limit[0])
            self.assertAlmostEqual(gt_limit[1], parsed_limit[1])


class ForwardKinematicTest(unittest.TestCase):
    def setUp(self):
        self.panda = PandaArm()

    def test_revolute_chain(self):
        """_summary_"""
        device = "cuda"
        joint_values = torch.rand((10, 7), device=device)
        t, R, runtime = self.panda.forward_kinematics_batch(joint_values, device)
        # TODO: Compare against ground truth FK data

    def test_continuous_chain(self):
        """_summary_"""
        pass


if __name__ == "__main__":
    unittest.main()
