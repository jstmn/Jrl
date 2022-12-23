import unittest

import torch
import numpy as np

from jkinpylib.conversions import *
from jkinpylib.utils import set_seed

# Set seed to ensure reproducibility
set_seed()

# suppress=True: print with decimal notation, not scientific
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True)


class TestSolutionRerfinement(unittest.TestCase):
    def test_quaternion_to_rpy_batch(self):
        """Test quaternion_to_rpy_pt"""
        pass

    def test_quaternion_conjugate_batch_np(self):
        # w, x, y, z
        q0 = np.array([[1, 0, 0, 0], [0.7071068, 0, 0, 0.7071068]])  # 90 deg rotation about +z
        q0_conjugate_expected = np.array([[1, 0, 0, 0], [0.7071068, 0, 0, -0.7071068]])  # 90 deg rotation about +z
        q0_conjugate_returned = quaternion_conjugate_np(q0)
        self.assertEqual(q0_conjugate_returned.shape, (2, 4))
        np.testing.assert_almost_equal(q0_conjugate_returned, q0_conjugate_expected)

    def test_quaternion_norm(self):
        # w, x, y, z
        qs = np.array([[1, 0, 0, 0], [0.7071068, 0, 0, 0.7071068], [1.0, 1.0, 0, 0.0]])  # 90 deg rotation about +z
        norms_expected = np.array([1, 1, 1.414213562])
        norms_returned = quaternion_norm_np(qs)
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
