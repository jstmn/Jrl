import unittest

import torch
import numpy as np
import math

from jrl.utils import set_seed
from jrl.geometry import (
    capsule_capsule_distance_batch,
    capsule_cuboid_distance_batch,
    cuboid_sphere_distance_batch,
    sphere_capsule_distance_batch,
    CuboidUtils,
)
from jrl.math_utils import quaternion_to_rotation_matrix
from jrl.config import DEVICE

import fcl

# Set seed to ensure reproducibility
set_seed()
torch.set_default_dtype(torch.float32)
torch.set_default_device(DEVICE)


class TestGeometry(unittest.TestCase):
    def test_capsule_capsule_distance_batch(self):
        """Test capsule_capsule_distance_batch() returns the expected distance"""

        # Test 1: two height=2 capsules parallel to one another and pointing in +z. The distance between them should be
        # 1.5: their +z axis are 2 apart, each has a radius of .25, so the distance between their cylinder portions is
        # 1.5

        # Both are capsules with radius=0.25, height=2.0
        c1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.25]])
        c2 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.25]])

        T1 = torch.eye(4).unsqueeze(0)
        T1[:, 0, 3] = -1
        T2 = torch.eye(4).unsqueeze(0)
        T2[:, 0, 3] = 1

        returned = capsule_capsule_distance_batch(c1, T1, c2, T2).cpu()
        expected = 1.5

        np.testing.assert_allclose(returned, expected)

        np.random.seed(12345)
        for i in range(20):
            r1, r2, h1, h2 = np.random.uniform(0.1, 1.0, (4,))
            q1 = np.random.randn(4)
            q1 /= np.linalg.norm(q1)
            q2 = np.random.randn(4)
            q2 /= np.linalg.norm(q2)
            R1 = quaternion_to_rotation_matrix(torch.tensor(q1).unsqueeze(0)).cpu().numpy().reshape((3, 3))
            R2 = quaternion_to_rotation_matrix(torch.tensor(q2).unsqueeze(0)).cpu().numpy().reshape((3, 3))
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
            T1 = torch.eye(4).unsqueeze(0)
            T1[:, :3, :3] = torch.tensor(R1)
            T1[:, :3, 3] = torch.tensor(t1)
            T2 = torch.eye(4).unsqueeze(0)
            T2[:, :3, :3] = torch.tensor(R2)
            T2[:, :3, 3] = torch.tensor(t2)

            our_dist = capsule_capsule_distance_batch(c1, T1, c2, T2).cpu()
            np.testing.assert_allclose(fcl_dist, our_dist, atol=1e-4)

    def test_capsule_cuboid_distance_batch(self):
        """Test capsule_cuboid_distance_batch() returns the expected distance"""

        Tcaps = torch.eye(4).unsqueeze(0)
        Tcaps[:, :3, 3] = torch.tensor([2.0, 0.0, 0.0])
        Tcube = torch.eye(4).unsqueeze(0)

        caps = torch.tensor([[0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.25]])
        cube = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]])

        returned = capsule_cuboid_distance_batch(caps, Tcaps, cube, Tcube).cpu()
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
            R1 = quaternion_to_rotation_matrix(torch.tensor(q1).unsqueeze(0)).cpu().numpy().reshape((3, 3))
            R2 = quaternion_to_rotation_matrix(torch.tensor(q2).unsqueeze(0)).cpu().numpy().reshape((3, 3))
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

            caps = torch.tensor([[0.0, 0.0, -h / 2, 0.0, 0.0, h / 2, r]], dtype=torch.float32)
            cuboid = torch.tensor(
                [[-xspan / 2, -yspan / 2, -zspan / 2, xspan / 2, yspan / 2, zspan / 2]], dtype=torch.float32
            )
            T1 = torch.eye(4).unsqueeze(0)
            T1[:, :3, :3] = torch.tensor(R1)
            T1[:, :3, 3] = torch.tensor(t1)
            T2 = torch.eye(4).unsqueeze(0)
            T2[:, :3, :3] = torch.tensor(R2)
            T2[:, :3, 3] = torch.tensor(t2)

            our_dist = capsule_cuboid_distance_batch(caps, T1, cuboid, T2).cpu()

            if our_dist[0] < 0.0:
                # FCL doesn't report penetration distance here, it just returns
                # -1.0 for colliding geometry
                our_dist[0] = -1.0
            else:
                pos_dist_count += 1

            np.testing.assert_allclose(fcl_dist, our_dist, atol=1e-3)

        self.assertGreater(pos_dist_count, 0)
        print("pos_dist_count:", pos_dist_count)

    # python -m unittest tests/geometry_test.py TestGeometry.test__cuboid_corners_in_world_frame
    def test__cuboid_corners_in_world_frame(self):
        # tf 0: identity
        # tf 1: rotz(90 deg) + translation([1, 1, 1])
        tfs = torch.cat(
            [
                torch.eye(4).view(1, 4, 4),
                torch.tensor([
                    [0.0000000, -1.0000000, 0.0000000, 5.0],
                    [1.0000000, 0.0000000, 0.0000000, 5.0],
                    [0.0000000, 0.0000000, 1.0000000, 5.0],
                    [0, 0, 0, 1.0],
                ]).view(1, 4, 4),
            ],
            dim=0,
        )
        assert tfs.shape == (2, 4, 4)
        corners = torch.tensor([
            [-1, -1, -1, 2, 3, 4],  # ( x1, y1, z1, x2, y2, z2)
            [-2, -2, -2, 5.0, 1.0, 1.0],  # ( x1, y1, z1, x2, y2, z2)
        ])
        c0_expected = torch.tensor([[-1, -1, -1], [2 + 5.0, -2 + 5.0, -2 + 5.0]])
        c6_expected = torch.tensor([[2, 3, 4], [-1 + 5.0, 5 + 5.0, 1 + 5.0]])

        c0, c1, c2, c3, c4, c5, c6, c7 = CuboidUtils._cuboid_corners_in_world_frame(tfs, corners)

        self.assertEqual(c0_expected.shape, c0.shape)
        self.assertEqual(c6_expected.shape, c6.shape)
        torch.testing.assert_close(c0_expected, c0, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(c6_expected, c6, atol=1e-6, rtol=0.0)

    # python -m unittest tests.geometry_test.TestGeometry.test__get_cuboid_G_h
    def test__get_cuboid_G_h(self):

        # tf 0: identity
        # tf 1: rotz(90 deg) + translation([1, 1, 1])
        tfs = torch.cat(
            [
                torch.tensor([
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ]).view(1, 4, 4),
                torch.eye(4).view(1, 4, 4),
                torch.tensor([
                    [0.0000000, -1.0000000, 0.0000000, 3.0],
                    [1.0000000, 0.0000000, 0.0000000, 3.0],
                    [0.0000000, 0.0000000, 1.0000000, 2.0],
                    [0, 0, 0, 1.0],
                ]).view(1, 4, 4),
                torch.tensor([
                    [0.7071068, -0.5, 0.50, -1.0],
                    [0.5, 0.8535534, 0.1464466, -1.0],
                    [-0.5, 0.1464466, 0.8535534, 4.0],
                    [0, 0, 0, 1.0],
                ]).view(1, 4, 4),
            ],
            dim=0,
        )

        corners = torch.tensor([
            [-0.25, -0.25, -0.25, 0.25, 0.25, 0.25],
            [-1, -1, -1, 1, 0.5, 0.5],  # ( x1, y1, z1, x2, y2, z2)
            [-1, -1, -2, 0.5, 0.5, 1.0],
            [-1, -1, -1, 1.0, 1.0, 1.0],
        ])
        c0, c1, c2, c3, c4, c5, c6, c7 = CuboidUtils._cuboid_corners_in_world_frame(tfs, corners)
        G, h = CuboidUtils._get_cuboid_G_h(tfs, c0, c1, c2, c3, c4, c5, c6, c7, debug=False)
        self.assertEqual(G.shape, (len(tfs), 6, 3))  # G is [ n x 6 x 3]

        print()
        print("G: ", G)
        print("h: ", h)

        # Gx <= h
        # so should be that 'G*center <= h' / 'G*center - h <= 0'
        #   G[:, 0, :] = plane . lower
        #   G[:, 1, :] = plane . upper
        #   G[:, 2, :] = plane . left
        #   G[:, 3, :] = plane . right
        #   G[:, 4, :] = plane . front
        #   G[:, 5, :] = plane . back

        #
        #
        centers = tfs[:, :3, 3].view(len(tfs), 3, 1)
        res = G.bmm(centers) - h.view(len(tfs), 6, 1)
        print()
        print()
        print("===== Testing cube centers against constraints\n")
        print("centers:", centers)
        print("res:    ", res)
        self.assertLessEqual(res.max(), 0.0)
        print("SUCCESS - constraints satisfied for all cube centers")

        #
        #
        print()
        print()
        print("===== Testing point (0, 0, 0) against front plane constraint\n")
        G_0_front = G[0, 4]
        p_zero = torch.tensor([0.0, 0.0, 0.0])
        res_p_zero = G_0_front.dot(p_zero) - h[0, 4]
        assert res_p_zero.numel() == 1
        print("res_p_zero:", res_p_zero)

        # (0, 0., 0.0) is outside of the cube, so Gp - h should be positive
        print()
        print()
        print("===== Testing point outside of cube\n")
        G_0 = G[0]  # [ 6 x 3 ]
        p = torch.tensor([[0.0, 0.0, 0.0]]).T  # [ 3 x 1 ]
        res_2 = G_0.matmul(p.T.view(3, 1)).view(6) - h[0]
        assert res_2.numel() == 6, f"res.shape: {res.shape}"
        print("res_2:", res_2.view(6, 1))

        for i in range(6):
            constraint_names = ["lower", "upper", "left", "right", "front", "back"]
            if res_2[i] > 0.0:
                print(f"  {constraint_names[i]} constraint violated (res_2[{i}]: {res_2[i]})")

        self.assertGreater(res_2.max(), 0.0, f"one value must be positive, meaning there is a unsatisfied constraint")
        print("success - all constraints satisfied for point outside of cube")

    # python -m unittest tests.geometry_test.TestGeometry.test_sphere_cuboid
    def test_sphere_cuboid(self):
        # tf 0, 1: identity
        # tf 2: rotz(90 deg) + translation([1, 1, 1])
        tfs = torch.cat(
            [
                torch.eye(4).view(1, 4, 4),
                torch.eye(4).view(1, 4, 4),
                torch.tensor([
                    [0.0000000, -1.0000000, 0.0000000, 0.0],
                    [1.0000000, 0.0000000, 0.0000000, 0.0],
                    [0.0000000, 0.0000000, 1.0000000, 0.0],
                    [0, 0, 0, 1.0],
                ]).view(1, 4, 4),
                torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.5],
                    [0, 0, 0, 1.0],
                ]).view(1, 4, 4),
                torch.tensor([
                    [1.0000, 0.0000, 0.0000, 0.7874],
                    [0.0000, 1.0000, 0.0000, 0.9345],
                    [0.0000, 0.0000, 1.0000, 0.4503],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]).view(1, 4, 4),
            ],
            dim=0,
        )
        corners = torch.tensor([
            [-1, -1, -1, 1, 1, 1.0],
            [-1, -1, -1, 1, 1, 1.0],
            [-1, -1, -1, 1, 1, 1.0],
            [-1, -1, -1, 1, 1, 1.0],
            [-1, -1, -1, 1, 1, 1.0],
        ])

        sphere_centers = torch.tensor(
            [[2.0, 2.0, 2.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 1.5], [0.7874 + 1.0, 0.9345, 0.4503]]
        )
        sphere_radii = torch.tensor([[1.732050808 / 2], [0.25], [0.25], [0.25], [0.1]])

        # TODO: this is wrong i think, draw it out
        distances_expected = torch.tensor([[1.732050808 / 2], [0.75], [0.75], [0.75], [0.9]])
        distances_recv = cuboid_sphere_distance_batch(tfs, corners, sphere_centers, sphere_radii)
        self.assertEqual(distances_recv.numel(), len(tfs))
        print("distances_expected:", distances_expected)
        print("distances_recv:    ", distances_recv)
        torch.testing.assert_close(distances_expected, distances_recv, rtol=0.0, atol=0.001)

    # python -m unittest tests.geometry_test.TestGeometry.test_sphere_cuboid_specific
    def test_sphere_cuboid_specific(self):
        tfs = torch.cat(
            [
                torch.tensor([
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ]).view(1, 4, 4),
            ],
            dim=0,
        )
        corners = torch.tensor([
            [-0.25, -0.25, -0.25, 0.25, 0.25, 0.25],
        ])
        sphere_centers = torch.tensor([[0.0, 0.0, 0.0]])
        sphere_radii = torch.tensor([[0.1]])
        dist, sol, G, h = cuboid_sphere_distance_batch(tfs, corners, sphere_centers, sphere_radii, return_sol=True)

        print("dist:", dist)
        print("sol: ", sol)
        print("G:   ", G)
        print("h:   ", h)

        for i in range(6):
            constraint_names = ["lower", "upper", "left", "right", "front", "back"]
            constraint_val = G[0, i].dot(sol[0, :, 0]) - h[0, i]
            if constraint_val > 0.0:
                print(f"  {constraint_names[i]} constraint violated (res{i}: {constraint_val})")

    # python -m unittest tests.geometry_test.TestGeometry.test_sphere_capsule_distance_batch
    def test_sphere_capsule_distance_batch(self):

        # Capsules refrence frame have end points in the +x direction
        # capsule center points are (0, 0, -1), (0, 0, 1)
        capsules = torch.tensor([
            [0, 0, -1, 0, 0, 1, 0.1],
            [0, 0, -1, 0, 0, 1, 0.1],
            [0, 0, -1, 0, 0, 1, 0.1],
        ])
        capsule_poses = torch.eye(4).expand(capsules.shape[0], 4, 4)
        spheres = torch.tensor([
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 2.0, 0.2],
            [1.0, 1.0, 0.0, 0.2],  # dist from (0, 0, 1) to (1, 1, 2) minus 0.1, minus 0.25
        ])
        sqrt_2 = math.sqrt(2)
        distances_expected = torch.tensor([0.7, 0.7, sqrt_2 - 0.2 - 0.1])  # 1 - 0.2 - 0.1  # 1 - 0.2 - 0.1  #
        distances = sphere_capsule_distance_batch(capsules, capsule_poses, spheres)

        print("distances:         ", distances)
        print("\ndistances_expected:", distances_expected)
        # self.assertEqual(distances.numel(), 3)
        torch.testing.assert_close(distances, distances_expected, atol=0.001, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
