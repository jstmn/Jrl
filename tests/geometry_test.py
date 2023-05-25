import unittest

import torch
import numpy as np

from jkinpylib.geometry import capsule_capsule_distance_batch

from jkinpylib.utils import set_seed, to_torch
from jkinpylib.conversions import quaternion_to_rotation_matrix

import fcl

# Set seed to ensure reproducibility
set_seed()


class TestGeometry(unittest.TestCase):
    def test_capsule_capsule_distance_batch(self):
        """Test capsule_capsule_distance_batch() returns the expected distance"""

        # Test 1: two height=2 capsules parallel to one another and pointing in +z. The distance between them should be
        # 1.5: their +z axis are 2 apart, each has a radius of .25, so the distance between their cylinder portions is
        # 1.5

        # Both are capsules with radius=0.25, height=2.0
        c1 = torch.tensor([[0.25, 2.0]], device="cpu", dtype=torch.float32)
        c2 = torch.tensor([[0.25, 2.0]], device="cpu", dtype=torch.float32)

        T1 = torch.tensor([[-1, 0, 0, 1, 0, 0, 0]], device="cpu", dtype=torch.float32)
        T2 = torch.tensor([[1, 0, 0, 1, 0, 0, 0]], device="cpu", dtype=torch.float32)

        returned = capsule_capsule_distance_batch(c1, T1, c2, T2)
        expected = 1.5

        np.testing.assert_allclose(returned, expected)

        np.random.seed(12345)
        for i in range(20):
            r1, r2, h1, h2 = np.random.uniform(0.1, 1.0, (4,))
            q1 = np.random.randn(4)
            q1 /= np.linalg.norm(q1)
            q2 = np.random.randn(4)
            q2 /= np.linalg.norm(q2)
            R1 = quaternion_to_rotation_matrix(torch.tensor(q1).unsqueeze(0)).numpy().reshape((3, 3))
            R2 = quaternion_to_rotation_matrix(torch.tensor(q2).unsqueeze(0)).numpy().reshape((3, 3))
            t1 = np.random.randn(3)
            t2 = np.random.randn(3)

            fcl_c1 = fcl.Capsule(r1, h1)
            fcl_c2 = fcl.Capsule(r2, h2)
            fcl_T1 = fcl.Transform(R1, t1 + R1 @ np.array([0, 0, h1 / 2]))
            fcl_T2 = fcl.Transform(R2, t2 + R2 @ np.array([0, 0, h2 / 2]))
            o1 = fcl.CollisionObject(fcl_c1, fcl_T1)
            o2 = fcl.CollisionObject(fcl_c2, fcl_T2)

            request = fcl.DistanceRequest()
            result = fcl.DistanceResult()
            fcl_dist = fcl.distance(o1, o2, request, result)

            c1 = torch.tensor([[r1, h1]])
            c2 = torch.tensor([[r2, h2]])
            T1 = torch.cat((torch.tensor(t1), torch.tensor(q1)), dim=0).unsqueeze(0)
            T2 = torch.cat((torch.tensor(t2), torch.tensor(q2)), dim=0).unsqueeze(0)

            print(c1.dtype, T1.dtype, c2.dtype, T2.dtype)

            our_dist = capsule_capsule_distance_batch(c1, T1, c2, T2)

            print(fcl_dist)
            print(our_dist)
            np.testing.assert_allclose(fcl_dist, our_dist, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
