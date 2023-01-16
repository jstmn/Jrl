import unittest

from jkinpylib.urdf_utils import _len3_tuple_from_str
from jkinpylib.robots import get_all_robots, get_robot, Fetch, Panda

import torch
import numpy as np

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True)

ROBOTS = get_all_robots()


class RobotTest(unittest.TestCase):
    def test_jacobian_np(self):
        for robot in ROBOTS:
            x = np.zeros(robot.n_dofs)
            J = robot.jacobian_np(x)
            self.assertEqual(J.shape, (6, robot.n_dofs))

    def test_list_from_str(self):
        inputs = ["0 0 0.333", "0 0 1.0", "0 0 1", "1.5707963267948966   0 3.141592653589793"]
        expected_outputs = [(0, 0, 0.333), (0, 0, 1.0), (0, 0, 1.0), (1.5707963267948966, 0, 3.141592653589793)]
        for input_, expected_output in zip(inputs, expected_outputs):
            output = _len3_tuple_from_str(input_)
            self.assertIsInstance(output, tuple)
            self.assertEqual(len(output), 3)
            self.assertAlmostEqual(output[0], expected_output[0])
            self.assertAlmostEqual(output[1], expected_output[1])
            self.assertAlmostEqual(output[2], expected_output[2])

    def test_n_dofs(self):
        ground_truth_n_dofs = {
            "panda": 7,
            "baxter": 7,
            "fetch": 8,
            "iiwa7": 7,
        }
        for robot in ROBOTS:
            self.assertEqual(robot.n_dofs, ground_truth_n_dofs[robot.name])

    def test_joint_limits(self):
        ground_truth_joint_limits = {
            "fetch": [
                (0, 0.38615),
                (-1.6056, 1.6056),
                (-1.221, 1.518),  # shoulder_lift_joint
                (-np.pi, np.pi),  # upperarm_roll_joint
                (-2.251, 2.251),  # elbow_flex_joint
                (-np.pi, np.pi),  # forearm_roll_joint
                (-2.16, 2.16),  # wrist_flex_joint
                (-np.pi, np.pi),  # wrist_roll_joint
            ],
            "panda": [
                (-2.8973, 2.8973),
                (-1.7628, 1.7628),
                (-2.8973, 2.8973),
                (-3.0718, -0.0698),
                (-2.8973, 2.8973),
                (-0.0175, 3.7525),
                (-2.8973, 2.8973),
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
            "iiwa7": [
                (-2.9670597283903604, 2.9670597283903604),
                (-2.0943951023931953, 2.0943951023931953),
                (-2.9670597283903604, 2.9670597283903604),
                (-2.0943951023931953, 2.0943951023931953),
                (-2.9670597283903604, 2.9670597283903604),
                (-2.0943951023931953, 2.0943951023931953),
                (-3.0543261909900763, 3.0543261909900763),
            ],
        }
        for robot in ROBOTS:
            self.assertEqual(len(robot.actuated_joints_limits), robot.n_dofs)
            for gt_limit, parsed_limit in zip(ground_truth_joint_limits[robot.name], robot.actuated_joints_limits):
                self.assertAlmostEqual(gt_limit[0], parsed_limit[0])
                self.assertAlmostEqual(gt_limit[1], parsed_limit[1])

    def test_actuated_joint_names(self):
        ground_truth_actuated_joints = {
            "panda": [
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
            "fetch": [
                "torso_lift_joint",
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",  # continuous
                "elbow_flex_joint",
                "forearm_roll_joint",  # continuous
                "wrist_flex_joint",
                "wrist_roll_joint",  # continous
            ],
            "iiwa7": [
                "iiwa_joint_1",
                "iiwa_joint_2",
                "iiwa_joint_3",
                "iiwa_joint_4",
                "iiwa_joint_5",
                "iiwa_joint_6",
                "iiwa_joint_7",
            ],
        }
        for robot in ROBOTS:
            self.assertEqual(len(robot.actuated_joint_names), robot.n_dofs)
            self.assertListEqual(robot.actuated_joint_names, ground_truth_actuated_joints[robot.name])

    def test_q_x_conversion(self):
        for robot in ROBOTS:
            ndofs = robot.n_dofs
            x_original = np.array(
                [
                    [0] * ndofs,
                    list(i * 0.1 for i in range(ndofs)),
                ]
            )
            q = robot._x_to_qs(x_original)
            x_returned = robot._qs_to_x(q)
            np.testing.assert_allclose(x_original, x_returned)

    def test_x_driver_vec_conversion_panda(self):
        gt_klampt_vector_dimensionality = {
            "fetch": 14,  # 24 total joints, 10 of them are fixed
            "panda": 8,  # panda has 1 non user specified actuated joint
            "iiwa7": 7,  #
        }

        for robot in ROBOTS:
            gt_vector_dim = gt_klampt_vector_dimensionality[robot.name]
            ndof = robot.n_dofs

            # test 1
            x_original = np.zeros(ndof)
            driver_vec = robot._driver_vec_from_x(x_original)
            self.assertEqual(len(driver_vec), gt_vector_dim)
            x_returned = robot._x_from_driver_vec(driver_vec)
            np.testing.assert_allclose(x_original, x_returned)

            # test 2
            x_original = 0.1 * np.arange(ndof)
            driver_vec = robot._driver_vec_from_x(x_original)
            self.assertEqual(len(driver_vec), gt_vector_dim)  # panda has 1 non user specified actuated joint
            x_returned = robot._x_from_driver_vec(driver_vec)
            np.testing.assert_allclose(x_original, x_returned)


if __name__ == "__main__":
    unittest.main()
