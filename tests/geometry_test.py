import unittest

import torch
import numpy as np

from jkinpylib.geometry import (
    capsule_capsule_distance_batch,
    capsule_cuboid_distance_batch,
)

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
        c1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.25]], device="cpu", dtype=torch.float32)
        c2 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.25]], device="cpu", dtype=torch.float32)

        T1 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        T1[:, 0, 3] = -1
        T2 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        T2[:, 0, 3] = 1

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
            fcl_T1 = fcl.Transform(R1, t1)
            fcl_T2 = fcl.Transform(R2, t2)
            o1 = fcl.CollisionObject(fcl_c1, fcl_T1)
            o2 = fcl.CollisionObject(fcl_c2, fcl_T2)

            request = fcl.DistanceRequest()
            result = fcl.DistanceResult()
            fcl_dist = fcl.distance(o1, o2, request, result)

            c1 = torch.tensor([[0.0, 0.0, -h1 / 2, 0.0, 0.0, h1 / 2, r1]], dtype=torch.float32)
            c2 = torch.tensor([[0.0, 0.0, -h2 / 2, 0.0, 0.0, h2 / 2, r2]], dtype=torch.float32)
            T1 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            T1[:, :3, :3] = torch.tensor(R1)
            T1[:, :3, 3] = torch.tensor(t1)
            T2 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            T2[:, :3, :3] = torch.tensor(R2)
            T2[:, :3, 3] = torch.tensor(t2)

            print(c1.dtype, T1.dtype, c2.dtype, T2.dtype)

            our_dist = capsule_capsule_distance_batch(c1, T1, c2, T2)

            print(fcl_dist)
            print(our_dist)
            np.testing.assert_allclose(fcl_dist, our_dist, atol=1e-4)

    def test_capsule_cuboid_distance_batch(self):
        """Test capsule_cuboid_distance_batch() returns the expected distance"""

        Tcaps = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        Tcaps[:, :3, 3] = torch.tensor([2.0, 0.0, 0.0])
        Tcube = torch.eye(4, dtype=torch.float32).unsqueeze(0)

        caps = torch.tensor([[0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.25]], dtype=torch.float32)
        cube = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

        returned = capsule_cuboid_distance_batch(caps, Tcaps, cube, Tcube)
        expected = 0.75

        np.testing.assert_allclose(returned, expected, atol=1e-4)

        np.random.seed(12345)
        pos_dist_count = 0
        for i in range(50):
            r, h = np.random.uniform(0.1, 1.0, (2,))
            xspan, yspan, zspan = np.random.uniform(0.1, 1.0, (3,))
            q1 = np.random.randn(4)
            q1 /= np.linalg.norm(q1)
            q1 = np.array([1.0, 0.0, 0.0, 0.0])
            q2 = np.random.randn(4)
            q2 /= np.linalg.norm(q2)
            q2 = np.array([1.0, 0.0, 0.0, 0.0])
            R1 = quaternion_to_rotation_matrix(torch.tensor(q1).unsqueeze(0)).numpy().reshape((3, 3))
            R2 = quaternion_to_rotation_matrix(torch.tensor(q2).unsqueeze(0)).numpy().reshape((3, 3))
            t1 = np.random.randn(3)
            t2 = np.random.randn(3)

            fcl_caps = fcl.Capsule(r, h)
            fcl_Tcaps = fcl.Transform(R1, t1)
            o_caps = fcl.CollisionObject(fcl_caps, fcl_Tcaps)

            fcl_box = fcl.Box(xspan, yspan, zspan)
            fcl_Tbox = fcl.Transform(R2, t2)
            o_box = fcl.CollisionObject(fcl_box, fcl_Tbox)

            request = fcl.DistanceRequest()
            result = fcl.DistanceResult()
            fcl_dist = fcl.distance(o_caps, o_box, request, result)

            caps = torch.tensor(
                [[0.0, 0.0, -h / 2, 0.0, 0.0, h / 2, r]],
                dtype=torch.float32,
            )
            cuboid = torch.tensor(
                [[-xspan / 2, -yspan / 2, -zspan / 2, xspan / 2, yspan / 2, zspan / 2]],
                dtype=torch.float32,
            )
            T1 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            T1[:, :3, :3] = torch.tensor(R1)
            T1[:, :3, 3] = torch.tensor(t1)
            T2 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            T2[:, :3, :3] = torch.tensor(R2)
            T2[:, :3, 3] = torch.tensor(t2)

            our_dist = capsule_cuboid_distance_batch(caps, T1, cuboid, T2)

            if our_dist[0] < 0.0:
                # FCL doesn't report penetration distance here, it just returns
                # -1.0 for colliding geometry
                our_dist[0] = -1.0
            else:
                pos_dist_count += 1

            np.testing.assert_allclose(fcl_dist, our_dist, atol=1e-3)

        self.assertGreater(pos_dist_count, 0)
        print("pos_dist_count:", pos_dist_count)


if __name__ == "__main__":
    unittest.main()
