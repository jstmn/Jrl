import unittest

import torch
import numpy as np

from jkinpylib.conversions import (
    quaternion_inverse,
    quaternion_to_rotation_matrix,
    geodesic_distance_between_rotation_matrices,
    quaternion_conjugate,
    quaternion_norm,
)
from jkinpylib.utils import set_seed

# Set seed to ensure reproducibility
set_seed()

# suppress=True: print with decimal notation, not scientific
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True)


class TestSolutionRerfinement(unittest.TestCase):
    def test_enforce_pt_np_input_decorator(self):
        """Test that the enforce_pt_np_input() decorator works as expected."""

        # Test 1: Catches non-tensor, non-numpy input
        quats = [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
        with self.assertRaises(AssertionError):
            quaternion_inverse(quats)

        # Test 2: Catches three arguments
        quats = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        with self.assertRaises(AssertionError):
            quaternion_inverse(quats, quats, quats)

        # Test 3: Checks that both inputs are the same type
        quats_pt = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        quats_np = np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        with self.assertRaises(AssertionError):
            quaternion_inverse(quats_pt, quats_np)

    def test_geodesic_distance(self):
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

    def test_quaternion_conjugate_batch_np(self):
        # w, x, y, z
        q0 = np.array([[1, 0, 0, 0], [0.7071068, 0, 0, 0.7071068]])  # 90 deg rotation about +z
        q0_conjugate_expected = np.array([[1, 0, 0, 0], [0.7071068, 0, 0, -0.7071068]])  # 90 deg rotation about +z
        q0_conjugate_returned = quaternion_conjugate(q0)
        self.assertEqual(q0_conjugate_returned.shape, (2, 4))
        np.testing.assert_almost_equal(q0_conjugate_returned, q0_conjugate_expected)

    def test_quaternion_norm(self):
        # w, x, y, z
        qs = np.array([[1, 0, 0, 0], [0.7071068, 0, 0, 0.7071068], [1.0, 1.0, 0, 0.0]])  # 90 deg rotation about +z
        norms_expected = np.array([1, 1, 1.414213562])
        norms_returned = quaternion_norm(qs)
        self.assertEqual(norms_returned.shape, (3,))
        np.testing.assert_almost_equal(norms_returned, norms_expected)

    def test_quaternion_multiply_np(self):
        q_1s = np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0.7071068, 0, 0, 0.7071068],  # 90 deg rotation about +z
                [0, 0.7071068, 0, 0.7071068],  # 90 deg rotation about +y
            ]
        )

        q_2s = np.array(
            [
                [1, 0, 0, 0],
                [0.7071068, 0, 0, 0.7071068],  # 90 deg rotation about +z
                [0.7071068, 0, 0, 0.7071068],  # 90 deg rotation about +z
                [0.7071068, 0, 0, 0.7071068],  # 90 deg rotation about +z
            ]
        )

        # ground truth from https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/arithmetic/index.htm
        product_expected = np.array(
            [
                [1, 0, 0, 0],
                [0.7071068, 0, 0, 0.7071068],
                [0, 0, 0, 1],
                [-0.5, 0.5, -0.5, 0.5],
            ]
        )


if __name__ == "__main__":
    unittest.main()
