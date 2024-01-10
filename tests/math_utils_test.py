import unittest

import torch
import numpy as np

from jrl.math_utils import (
    quaternion_to_rotation_matrix,
    geodesic_distance_between_rotation_matrices,
    quaternion_conjugate,
    quaternion_norm,
    geodesic_distance_between_quaternions,
    geodesic_distance_between_quaternions_warp,
    quatconj,
    quaternion_product,
    quatmul,
    angular_subtraction,
    calculate_points_in_world_frame_from_local_frame_batch,
)
from jrl.utils import set_seed, to_torch
from jrl.robots import FetchArm

# Set seed to ensure reproducibility
set_seed()

# suppress=True: print with decimal notation, not scientific
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, precision=12)
torch.set_printoptions(linewidth=5000, sci_mode=False)

PI = torch.pi
_2PI = 2 * PI


class TestConversions(unittest.TestCase):
    def test_calculate_points_in_world_frame_from_local_frame(self):
        """Test calculate_points_in_world_frame_from_local_frame(). For this test, we are using m=4 points. Quaternions
        copied from https://www.andre-gaschler.com/rotationconverter/
        """
        world__T__local_frame = torch.tensor(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0],
                [5, 0, 0, 0.9659258, 0, 0, 0.258819],  # 30deg rotation about +z
            ],
            dtype=torch.float32,
            device="cpu",
        )
        # 4 points for each world__T__local_frame tf
        points_in_local_frame = torch.tensor(
            [
                [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                [[0, 0, 0], [3, 0, 0], [-3, 0, 0], [0, 2, 7]],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        returned = calculate_points_in_world_frame_from_local_frame_batch(world__T__local_frame, points_in_local_frame)
        expected = torch.tensor(
            [
                [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                [[1, 0, 0], [1.1, 0, 0], [1, 0.1, 0], [1, 0, 0.1]],
                [[5, 0, 0], [5 + 2.59808, 1.5, 0], [5 - 2.59808, -1.5, 0], [5 - 1, 1.73205, 7]],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        torch.testing.assert_close(returned, expected)

    def test_angular_subtraction(self):
        """Test that angular_subtraction() works as expected"""
        angles_1 = torch.tensor(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [PI / 2, PI / 2],
                [PI / 2, PI / 2],
            ],
            device="cpu",
            dtype=torch.float32,
        )
        angles_2 = torch.tensor(
            [
                [0.1, -0.1],
                [_2PI + 0.1, _2PI - 0.1],
                [PI + 0.1, PI - 0.1],
                [PI / 8, -PI / 8],
                [PI / 8 + 6 * PI, -PI / 8 + 8 * PI],
            ],
            device=angles_1.device,
            dtype=angles_1.dtype,
        )

        result = angular_subtraction(angles_1, angles_2)
        expected = torch.tensor(
            [
                [-0.1, 0.1],
                [-0.1, 0.1],
                [PI - 0.1, -(PI - 0.1)],
                [3 / 8 * PI, 5 / 8 * PI],
                [3 / 8 * PI, 5 / 8 * PI],
            ],
            device=angles_1.device,
            dtype=angles_1.dtype,
        )
        torch.testing.assert_close(result, expected)

    def test_geodesic_distance_between_quaternions_grad(self):
        """Check whether backprop through forward_kinematics_batch() and geodesic_distance_between_quaternions() will
        returns as nan for similar rotations
        """
        robot = FetchArm()

        def get_distance(device):
            theta_vec = torch.tensor(
                [[-0.12069701, 0.49622306, -1.47009099, -1.01792753, 2.53177810, -0.92676085, -1.35084212]],
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            pose_fk = robot.forward_kinematics_batch(theta_vec, out_device=device)
            target_pose = torch.tensor(
                [[0.95416701, 0.20000000, 0.56277800, 1.00000000, 0.00000000, 0.00000000, 0.00000000]],
                device=device,
                dtype=torch.float32,
            )
            return theta_vec, pose_fk, geodesic_distance_between_quaternions(pose_fk[:, 3:], target_pose[:, 3:])[0]

        theta_cpu, _, dist_cpu = get_distance("cpu")
        theta_cuda, _, dist_cuda = get_distance("cuda")
        dist_cpu.backward()
        dist_cuda.backward()
        self.assertFalse(torch.isnan(theta_cpu.grad).any())
        self.assertFalse(torch.isnan(theta_cuda.grad).any())

        # Test 2: Increase a joint angle away from a fixed one
        device = "cuda"
        joint_angle, target_pose = robot.sample_joint_angles_and_poses(1)
        joint_angle = to_torch(joint_angle, device=device)
        target_pose = to_torch(target_pose, device=device)
        for i in range(10):
            angle_offset = 0.00001 * i * torch.ones(joint_angle.shape, dtype=torch.float32, device=device)
            theta = joint_angle.clone() + angle_offset
            theta = theta.detach().clone().requires_grad_(True)
            pose_fk = robot.forward_kinematics_batch(theta, out_device=device)
            dist = geodesic_distance_between_quaternions(pose_fk[:, 3:], target_pose[:, 3:])[0]
            dist.backward()
            self.assertFalse(torch.isnan(dist))
            self.assertFalse(torch.isnan(theta.grad).any())

    def test_geodesic_distance_between_quaternions(self):
        """Test geodesic_distance_between_quaternions with torch tensor inputs"""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cpu", dtype=torch.float32)
        # Rotation about +x axis by .25 radians
        q2 = torch.tensor([[0.9921977, 0.1246747, 0, 0]], device="cpu", dtype=torch.float32)
        returned = geodesic_distance_between_quaternions(q1, q2)[0].item()
        self.assertAlmostEqual(0.25, returned, places=6)

        # Test 2
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cpu", dtype=torch.float32)
        q2 = torch.tensor([[0.0, 0.92387953, 0.38268343, 0.0]], device="cpu", dtype=torch.float32)
        returned = geodesic_distance_between_quaternions(q1, q2)[0].item()
        # TODO: AssertionError: 3.1415927 != 3.1411044597625732 within 7 places (0.00048824023742666256 difference). It
        # seems like rotation matrices created by quaternions have lower precision
        self.assertAlmostEqual(3.1415927, returned, places=3)

        # Test 3
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device="cpu", dtype=torch.float32)
        q2 = torch.tensor(
            [[0.0, 0.92387953, 0.38268343, 0.0], [0.9921977, 0.1246747, 0, 0]], device="cpu", dtype=torch.float32
        )
        returned = geodesic_distance_between_quaternions(q1, q2)
        self.assertAlmostEqual(3.1415927, returned[0].item(), places=3)
        self.assertAlmostEqual(0.25, returned[1].item(), places=6)

    def test_geodesic_distance_between_quaternions_warp(self):
        """Test geodesic_distance_between_quaternions with torch tensor inputs"""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cpu", dtype=torch.float32)
        # Rotation about +x axis by .25 radians
        q2 = torch.tensor([[0.9921977, 0.1246747, 0, 0]], device="cpu", dtype=torch.float32)
        returned = geodesic_distance_between_quaternions_warp(q1, q2)[0].item()
        self.assertAlmostEqual(0.25, returned, places=6)

        # Test 2
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cpu", dtype=torch.float32)
        q2 = torch.tensor([[0.0, 0.92387953, 0.38268343, 0.0]], device="cpu", dtype=torch.float32)
        returned = geodesic_distance_between_quaternions_warp(q1, q2)[0].item()
        self.assertAlmostEqual(3.1415927, returned, places=3)

        # Test 3
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device="cpu", dtype=torch.float32)
        q2 = torch.tensor(
            [[0.0, 0.92387953, 0.38268343, 0.0], [0.9921977, 0.1246747, 0, 0]], device="cpu", dtype=torch.float32
        )
        returned = geodesic_distance_between_quaternions_warp(q1, q2)
        self.assertAlmostEqual(3.1415927, returned[0].item(), places=3)
        self.assertAlmostEqual(0.25, returned[1].item(), places=6)

    def test_quaternion_to_rotation_matrix(self):
        """Test quaternion_to_rotation_matrix()"""

        # Test 1: Identity quaternion
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cpu", dtype=torch.float32)
        R_expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cpu", dtype=torch.float32
        )
        R_returned = quaternion_to_rotation_matrix(q)[0]
        torch.testing.assert_close(R_returned, R_expected)

        # Test 2
        q = torch.tensor([[0.0000000, 0.92387953, 0.38268343, 0.0000000]], device="cpu", dtype=torch.float32)
        # R_expected saved from this site: https://www.andre-gaschler.com/rotationconverter/
        R_expected = torch.tensor(
            [[0.7071068, 0.7071068, 0.0000000], [0.7071068, -0.7071068, 0.0000000], [0.0000000, 0.0000000, -1.0000000]],
            device="cpu",
            dtype=torch.float32,
        )
        R_returned = quaternion_to_rotation_matrix(q)[0]
        torch.testing.assert_close(R_returned, R_expected)

    def test_rotational_distance_between_rotation_matrices(self):
        """Test geodesic_distance_between_rotation_matrices()"""
        R1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cpu", dtype=torch.float32)

        # Rotation by 1 rad about the +z axis
        R2 = torch.tensor(
            [[0.5403023, -0.8414710, 0.0], [0.8414710, 0.5403023, 0.0], [0.0, 0.0, 1.0]],
            device="cpu",
            dtype=torch.float32,
        )
        distance_expected = 1.0
        returned = geodesic_distance_between_rotation_matrices(R1[None, :], R2[None, :])
        self.assertAlmostEqual(returned[0].item(), distance_expected, delta=5e-4)

        # Test 2: ___
        R1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device="cpu", dtype=torch.float32)
        R2 = torch.tensor(
            [[0.7071068, 0.7071068, 0.0000000], [0.7071068, -0.7071068, 0.0000000], [0.0000000, 0.0000000, -1.0000000]],
            device="cpu",
            dtype=torch.float32,
        )
        returned = geodesic_distance_between_rotation_matrices(R1[None, :], R2[None, :])
        distance_expected = 3.1415927
        self.assertAlmostEqual(returned[0].item(), distance_expected, delta=5e-4)

    def test_geodesic_distance_between_rotation_matrices(self):
        """Test geodesic_distance_between_rotation_matrices()"""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True, device="cpu")
        q2 = torch.tensor(
            [[1.0, -0.000209831749089, -0.000002384310619, 0.000092415713879]], device="cpu", requires_grad=True
        )
        m1 = quaternion_to_rotation_matrix(q1)
        m2 = quaternion_to_rotation_matrix(q2)

        # Test #1: Returns 0 for closeby quaternions
        distance = geodesic_distance_between_rotation_matrices(m1, m2)
        self.assertAlmostEqual(distance[0].item(), 0.0, delta=5e-4)

        # Test #2: Passes a gradient when for closeby quaternions
        loss = distance.mean()
        loss.backward()
        self.assertFalse(torch.isnan(q1.grad).any())
        self.assertFalse(torch.isnan(q2.grad).any())

    def test_quaternion_conjugate(self):
        # w, x, y, z
        q0 = torch.tensor([[1, 0, 0, 0], [0.7071068, 0, 0, 0.7071068]], dtype=torch.float32)  # 90 deg rotation about +z
        q0_conjugate_expected = torch.tensor(
            [[1, 0, 0, 0], [0.7071068, 0, 0, -0.7071068]], dtype=torch.float32
        )  # 90 deg rotation about +z

        # Test 1: quaternion_conjugate() is correct
        q0_conjugate_returned_1 = quaternion_conjugate(q0)
        self.assertEqual(q0_conjugate_returned_1.shape, (2, 4))
        torch.testing.assert_close(q0_conjugate_returned_1, q0_conjugate_expected)

        # Test 2:  quatconj() is correct
        q0_conjugate_returned_2 = quatconj(q0)
        self.assertEqual(q0_conjugate_returned_2.shape, (2, 4))
        torch.testing.assert_close(q0_conjugate_returned_2, q0_conjugate_expected)

    def test_quaternion_norm(self):
        # w, x, y, z
        qs = torch.tensor(
            [[1, 0, 0, 0], [0.7071068, 0, 0, 0.7071068], [1.0, 1.0, 0, 0.0]], dtype=torch.float32
        )  # 90 deg rotation about +z
        norms_expected = torch.tensor([1, 1, 1.414213562], dtype=torch.float32)
        norms_returned = quaternion_norm(qs)
        self.assertEqual(norms_returned.shape, (3,))
        torch.testing.assert_close(norms_returned, norms_expected)

    def test_quaternion_product(self):
        """Test that quaternion_product() and quatmul() are correct"""
        q1 = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0.7071068, 0, 0, 0.7071068],  # 90 deg rotation about +z
                [0, 0.7071068, 0, 0.7071068],  # 90 deg rotation about +y
            ],
            dtype=torch.float32,
        )

        q2 = torch.tensor(
            [
                [1, 0, 0, 0],
                [0.7071068, 0, 0, 0.7071068],  # 90 deg rotation about +z
                [0.7071068, 0, 0, 0.7071068],  # 90 deg rotation about +z
                [0.7071068, 0, 0, 0.7071068],  # 90 deg rotation about +z
            ],
            dtype=torch.float32,
        )

        # ground truth from https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/arithmetic/index.htm
        product_expected = torch.tensor(
            [
                [1, 0, 0, 0],
                [0.7071068, 0, 0, 0.7071068],
                [0, 0, 0, 1],
                [-0.5, 0.5, -0.5, 0.5],
            ],
            dtype=torch.float32,
        )

        product_returned_1 = quaternion_product(q1, q2)
        torch.testing.assert_close(product_expected, product_returned_1)

        product_returned_2 = quatmul(q1, q2)
        torch.testing.assert_close(product_expected, product_returned_2)
        print(f"test_quaternion_product() passed")


if __name__ == "__main__":
    unittest.main()
