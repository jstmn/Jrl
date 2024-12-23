from typing import Optional, Tuple
from time import time

import torch

import meshcat

from jrl.math_utils import QP
from jrl import meshcat_utils
from jrl.utils import evenly_spaced_colors


def capsule_capsule_distance_batch(
    caps1: torch.Tensor,
    T1: torch.Tensor,
    caps2: torch.Tensor,
    T2: torch.Tensor,
    use_qpth=False,
) -> float:
    """Returns the minimum distance between any two points on the given batch of capsules

    This function implements the capsule-capsule minimum distance equation from the paper "Efficient Calculation of
    Minimum Distance Between Capsules and Its Use in Robotics"
    (https://hal.science/hal-02050431/document).

    Each capsule is defined by two points in local frame, and a radius.
    The memory layout is [nx7]: [x1, y1, z1, x2, y2, z2, r1].

    Args:
        caps1 (torch.Tensor): [n x 7] tensor describing a batch of capsules.
        T1 (torch.Tensor): [n x 4 x 4] tensor describing the pose of the caps1 capsules
        caps2 (torch.Tensor): [n x 7] tensor describing a batch of capsules.
        T2 (torch.Tensor): [n x 4 x 4] tensor describing the pose of the caps1 capsules

    Returns:
        float: [n x 1] tensor with the minimum distance between each n capsules
    """
    dtype = caps1.dtype
    device = caps1.device

    n = caps1.shape[0]
    assert T1.shape == T2.shape == (n, 4, 4)
    assert caps1.shape == caps2.shape == (n, 7)

    # Local points are at the origin and top of capsule along the +z axis.
    r1 = caps1[:, 6]
    c1_world1 = T1[:, :3, :3].bmm(caps1[:, 0:3].unsqueeze(2)).squeeze(2) + T1[:, :3, 3]
    c1_world2 = T1[:, :3, :3].bmm(caps1[:, 3:6].unsqueeze(2)).squeeze(2) + T1[:, :3, 3]

    r2 = caps2[:, 6]
    c2_world1 = T2[:, :3, :3].bmm(caps2[:, 0:3].unsqueeze(2)).squeeze(2) + T2[:, :3, 3]
    c2_world2 = T2[:, :3, :3].bmm(caps2[:, 3:6].unsqueeze(2)).squeeze(2) + T2[:, :3, 3]

    p1 = c1_world1
    s1 = c1_world2 - p1
    p2 = c2_world1
    s2 = c2_world2 - p2

    A = torch.stack((s2, -s1), dim=2)
    y = (p2 - p1).unsqueeze(2)

    # Construct the QP
    Q = A.transpose(1, 2).bmm(A)
    # Semidefiniteness arises with parallel capsules
    Q = Q + 1e-4 * torch.eye(2, dtype=dtype, device=device).expand(n, -1, -1)
    p = 2 * A.transpose(1, 2).bmm(y).squeeze(2)

    # Inequality constraints
    G = torch.tensor(([[1, 0], [0, 1], [-1, 0], [0, -1]]), dtype=dtype, device=device).expand(n, -1, -1)
    h = torch.tensor([1, 1, 0, 0], dtype=dtype, device=device).expand(n, -1)

    # Solve the QP
    if use_qpth:
        import qpth

        e = torch.Tensor()  # Dummy equality constraint
        sol = qpth.qp.QPFunction(verbose=False)(2 * Q, p, G, h, e, e)
        sol = sol.unsqueeze(2)
    else:
        qp = QP(2 * Q, p, G, h)
        sol = qp.solve().unsqueeze(2)

    dist = torch.norm(A.bmm(sol) + y, dim=1) - r1.unsqueeze(1) - r2.unsqueeze(1)

    return dist


def capsule_cuboid_distance_batch(
    caps: torch.Tensor,
    Tcaps: torch.Tensor,
    cuboids: torch.Tensor,
    Tcuboids: torch.Tensor,
    use_qpth=False,
) -> torch.Tensor:
    """
    Returns the minimum distance between any two points on the given batches of
    capsules and cuboids.

    Args:
        caps (torch.Tensor): [n x 7] tensor descibing a batch of capsules.
        Tcaps (torch.Tensor): [n x 4 x 4] tensor describing the pose of the caps
        capsules.
        cuboids (torch.Tensor): [n x 6] (x1, y1, z1, x2, y2, z2) tensor
        describing a batch of cuboids, where (x1, y1, z1) is the bottom left
        corner and (x2, y2, z2) is the top right corner.
        Tcuboids (torch.Tensor): [n x 4 x 4] tensor describing the pose of the
        cuboids.

    Returns:
        torch.Tensor: [n x 1] tensor with the minimum distance between each n
        capsule and cuboid pair.
    """

    n = caps.shape[0]
    assert (
        Tcaps.shape == Tcuboids.shape == (n, 4, 4)
    ), f"Tcaps: {Tcaps.shape}, Tcuboids: {Tcuboids.shape}, Correct: ({n}, 4, 4)"
    assert caps.shape == (n, 7), f"{caps.shape}, ({n}, 7)"
    assert cuboids.shape == (n, 6), f"{cuboids.shape}, ({n}, 6)"

    device = caps.device
    dtype = caps.dtype

    # Put everything in cuboid frame
    r = caps[:, 6]
    p = Tcaps[:, :3, :3].bmm(caps[:, 0:3].unsqueeze(2)).squeeze(2) + Tcaps[:, :3, 3]
    p = Tcuboids[:, :3, :3].transpose(2, 1).bmm((p - Tcuboids[:, :3, 3]).unsqueeze(2)).squeeze(2)
    q = Tcaps[:, :3, :3].bmm(caps[:, 3:6].unsqueeze(2)).squeeze(2) + Tcaps[:, :3, 3]
    q = Tcuboids[:, :3, :3].transpose(2, 1).bmm((q - Tcuboids[:, :3, 3]).unsqueeze(2)).squeeze(2)
    s = q - p

    Q = torch.diag_embed(torch.ones(n, 4, dtype=dtype, device=device))
    Q[:, :3, 3] = -s
    Q[:, 3, :3] = -s
    Q[:, 3, 3] = (s * s).sum(dim=1)
    # Q = Q + 1e-4 * torch.eye(4, dtype=dtype, device=device).expand(n, -1, -1)
    Q = Q + 5e-4 * torch.eye(4, dtype=dtype, device=device).expand(n, -1, -1)

    p_ = torch.zeros(n, 4, dtype=dtype, device=device)
    p_[:, :3] = -2 * p
    p_[:, 3] = 2 * (s * p).sum(dim=1)

    G = torch.zeros(n, 8, 4, dtype=dtype, device=device)
    G[:, :4, :4] = -torch.eye(4, dtype=dtype, device=device)
    G[:, 4:, :4] = torch.eye(4, dtype=dtype, device=device)
    h = torch.zeros(n, 8, dtype=dtype, device=device)
    h[:, :3] = -cuboids[:, :3]
    h[:, 3] = 0
    h[:, 4:7] = cuboids[:, 3:]
    h[:, 7] = 1

    # Solve the QP
    if use_qpth:
        import qpth

        e = torch.Tensor()  # Dummy equality constraint
        sol = qpth.qp.QPFunction(verbose=False)(2 * Q, p_, G, h, e, e)
        sol = sol.unsqueeze(2)
    else:
        qp = QP(2 * Q, p_, G, h)
        sol = qp.solve().unsqueeze(2)

    dist = torch.norm(sol[:, :3, 0] - (s * sol[:, 3] + p), dim=1, keepdim=True) - r.unsqueeze(1)

    return dist


class CuboidUtils:
    @staticmethod
    def _cuboid_corners_in_world_frame(
        world__T__cuboids: torch.Tensor, cuboid_corners: torch.Tensor, visualize: bool = False
    ):
        """
        # TODO: do a more efficient version where do both bmm()'s at once.
        # NOTE: May be faster to add a dummy 1 vector to make the linalg workout and do `tfs.bmm(cuboid_corners[:, :3])[:, 0:3]`
        """
        n = cuboid_corners.shape[0]
        xmin, ymin, zmin, xmax, ymax, zmax = (
            cuboid_corners[:, 0].view(n, 1),  # FAILING HERE
            cuboid_corners[:, 1].view(n, 1),
            cuboid_corners[:, 2].view(n, 1),
            cuboid_corners[:, 3].view(n, 1),
            cuboid_corners[:, 4].view(n, 1),
            cuboid_corners[:, 5].view(n, 1),
        )
        c0 = torch.cat([xmin, ymin, zmin], dim=1).view(n, 3, 1)
        c1 = torch.cat([xmin, ymax, zmin], dim=1).view(n, 3, 1)
        c2 = torch.cat([xmax, ymax, zmin], dim=1).view(n, 3, 1)
        c3 = torch.cat([xmax, ymin, zmin], dim=1).view(n, 3, 1)
        c4 = torch.cat([xmin, ymin, zmax], dim=1).view(n, 3, 1)
        c5 = torch.cat([xmin, ymax, zmax], dim=1).view(n, 3, 1)
        c6 = torch.cat([xmax, ymax, zmax], dim=1).view(n, 3, 1)
        c7 = torch.cat([xmax, ymin, zmax], dim=1).view(n, 3, 1)

        t = world__T__cuboids[:, 0:3, 3]
        return (
            world__T__cuboids[:, 0:3, 0:3].bmm(c0).view(n, 3) + t,
            world__T__cuboids[:, 0:3, 0:3].bmm(c1).view(n, 3) + t,
            world__T__cuboids[:, 0:3, 0:3].bmm(c2).view(n, 3) + t,
            world__T__cuboids[:, 0:3, 0:3].bmm(c3).view(n, 3) + t,
            world__T__cuboids[:, 0:3, 0:3].bmm(c4).view(n, 3) + t,
            world__T__cuboids[:, 0:3, 0:3].bmm(c5).view(n, 3) + t,
            world__T__cuboids[:, 0:3, 0:3].bmm(c6).view(n, 3) + t,
            world__T__cuboids[:, 0:3, 0:3].bmm(c7).view(n, 3) + t,
        )

    @staticmethod
    def _get_cuboid_G_h(
        tfs: torch.Tensor,
        c0: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
        c6: torch.Tensor,
        c7: torch.Tensor,
        viz: Optional[meshcat.Visualizer] = None,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the G and h constraint matrices for a batch of cuboids

        Args:
            tfs (torch.Tensor) [ n x 4 x 4 ]:  world to cuboid frame transformations
            c0 (torch.Tensor) [ n x 3 ]:  list of c0 corners IN WORLD FRAME for the cuboids. See the docstring in
                                cuboid_sphere_distance_batch() for a description of each corner.
            c1 (torch.Tensor) [ n x 3 ]:  tensor. Same as c0, but for corner c1
            c2 (torch.Tensor) [ n x 3 ]: tensor. Same as c0, but for corner c2
            c3 (torch.Tensor) [ n x 3 ]: tensor. Same as c0, but for corner c3
            c4 (torch.Tensor) [ n x 3 ]: tensor. Same as c0, but for corner c4
            c5 (torch.Tensor) [ n x 3 ]: tensor. Same as c0, but for corner c5
            c6 (torch.Tensor) [ n x 3 ]: tensor. Same as c0, but for corner c6
            c7 (torch.Tensor) [ n x 3 ]: tensor. Same as c0, but for corner c7
            visualize (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: G and h
        """
        n = c0.shape[0]

        if debug:
            print()
            for i in range(n):
                print(f"cube {i}:")
                print("  c0: ", c0[i])
                print("  c1: ", c1[i])
                print("  c2: ", c2[i])
                print("  c3: ", c3[i])
                print("  c4: ", c4[i])
                print("  c5: ", c5[i])
                print("  c6: ", c6[i])
                print("  c7: ", c7[i])
                print()

        # 1. find the equations of the planes
        # Ax = b
        # A_lower: [ n x 3 x 3 ]
        # b is always 1. This means the equation to solve is a^T x = 1
        A_lower = torch.zeros((n, 3, 3))
        A_lower[:, 0, 0:3] = c0
        A_lower[:, 1, 0:3] = c2
        A_lower[:, 2, 0:3] = c3

        A_upper = torch.zeros((n, 3, 3))
        A_upper[:, 0, 0:3] = c5
        A_upper[:, 1, 0:3] = c6
        A_upper[:, 2, 0:3] = c7

        A_left = torch.zeros((n, 3, 3))
        A_left[:, 0, 0:3] = c5
        A_left[:, 1, 0:3] = c4
        A_left[:, 2, 0:3] = c0

        A_right = torch.zeros((n, 3, 3))
        A_right[:, 0, 0:3] = c7
        A_right[:, 1, 0:3] = c6
        A_right[:, 2, 0:3] = c2

        A_front = torch.zeros((n, 3, 3))
        A_front[:, 0, 0:3] = c0
        A_front[:, 1, 0:3] = c4
        A_front[:, 2, 0:3] = c7

        A_back = torch.zeros((n, 3, 3))
        A_back[:, 0, 0:3] = c6
        A_back[:, 1, 0:3] = c5
        A_back[:, 2, 0:3] = c1

        b = torch.ones((n, 3, 1))

        eps = 1e-6
        # TODO: solve one batched system instead of n batched systems
        plane_lower = torch.linalg.solve(A_lower + eps * torch.eye(3).expand(n, 3, 3), b).view(n, 3)  # [n x 3]
        plane_upper = torch.linalg.solve(A_upper + eps * torch.eye(3).expand(n, 3, 3), b).view(n, 3)
        plane_left = torch.linalg.solve(A_left + eps * torch.eye(3).expand(n, 3, 3), b).view(n, 3)
        plane_right = torch.linalg.solve(A_right + eps * torch.eye(3).expand(n, 3, 3), b).view(n, 3)
        plane_front = torch.linalg.solve(A_front + eps * torch.eye(3).expand(n, 3, 3), b).view(n, 3)
        plane_back = torch.linalg.solve(A_back + eps * torch.eye(3).expand(n, 3, 3), b).view(n, 3)

        if debug:
            print()
            for i in range(n):
                print(f"cube {i}:")
                print("  plane_lower:\ta:", plane_lower[i, 0:3], "\tb:", 1.0)
                print("  plane_upper:\ta:", plane_upper[i, 0:3], "\tb:", 1.0)
                print("  plane_left:\ta:", plane_left[i, 0:3], "\tb:", 1.0)
                print("  plane_right:\ta:", plane_right[i, 0:3], "\tb:", 1.0)
                print("  plane_front:\ta:", plane_front[i, 0:3], "\tb:", 1.0)
                print("  plane_back:\ta:", plane_back[i, 0:3], "\tb:", 1.0)
                print()

        if viz is not None:
            cube_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            plane_colors = evenly_spaced_colors(6)
            for i in range(n):
                pcloud_points = torch.cat([c0[i], c1[i], c2[i], c3[i], c4[i], c5[i], c6[i], c7[i]], dim=0).view(8, 3)
                meshcat_utils.add_pointcloud(viz, pcloud_points, color=cube_colors[i])

                assert torch.max(pcloud_points, dim=0).values.numel() == 3
                for j, plane in enumerate([plane_right, plane_back, plane_upper, plane_lower, plane_left, plane_front]):
                    assert plane[i].numel() == 3
                    meshcat_utils.add_plane(viz, plane[i, 0:3], 1.0, center_point=tfs[i, 0:3, 3], color=plane_colors[j])

        G = torch.zeros(n, 6, 3)  # [ n x 6 x 3 ]
        h = torch.ones(n, 6)

        G[:, 0, :] = plane_lower
        G[:, 1, :] = plane_upper
        G[:, 2, :] = plane_left
        G[:, 3, :] = plane_right
        G[:, 4, :] = plane_front
        G[:, 5, :] = plane_back

        centers = tfs[:, :3, 3].view(len(tfs), 3, 1)
        res = (G.bmm(centers) - h.view(n, 6, 1)).view(n, 6)

        # feasible set are all x that satisfy 'Gx <= h'. Therefore want 'G x_center - h <= 0'
        # if 'G x_center - h > 0', then set 'G = -G'
        G[res > 0] = -G[res > 0]
        h[res > 0] = -h[res > 0]
        return G, h


def cuboid_sphere_distance_batch(
    world__T__cuboids: torch.Tensor,
    cuboid_corners: torch.Tensor,
    sphere_centers: torch.Tensor,
    sphere_radii: torch.Tensor,
    use_qpth: bool = False,
    debug_timing: bool = False,
    return_sol: bool = False,
    viz: Optional[meshcat.Visualizer] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Returns the minimum distance between any two points on the given batches of capsules and cuboids.

    Args:
        world__T__cuboids (torch.Tensor): [n x 4 x 4] tensor describing the pose of the cuboids.
        cuboid_corners (torch.Tensor): [n x 6] (x1, y1, z1, x2, y2, z2) tensor describing a batch of cuboids, where
                                (x1, y1, z1) is the bottom left corner and (x2, y2, z2) is the top right corner. The
                                corners are in cuboid frame.

    Returns:
        torch.Tensor: [n x 1] tensor with the minimum distance between each capsule and cuboid pair.
    """
    assert len({x.shape[0] for x in [cuboid_corners, world__T__cuboids, sphere_centers, sphere_radii]}) == 1, (
        "different batch sizes detected"
        f" {cuboid_corners.shape[0], world__T__cuboids.shape[0], sphere_centers.shape[0], sphere_radii.shape[0]}"
    )
    n = world__T__cuboids.shape[0]
    assert world__T__cuboids.shape == (n, 4, 4), f"world__T__cuboids: {world__T__cuboids.shape}, should be: ({n}, 4, 4)"
    assert sphere_centers.shape == (n, 3), f"sphere centers are {sphere_centers.shape}, should be ({n}, 7)"
    assert cuboid_corners.shape == (n, 6), f"shape is {cuboid_corners.shape}, should be ({n}, 6)"
    assert sphere_radii.shape == (n, 1), f"shape is {sphere_radii.shape}, should be ({n}, 1)"

    """ G, h formulation

            c5                c6=(x2, y2, z2)
            +-----------------+
           /                 /|
          /                 / |
      c4 /              c7 /  |
        +-----------------+   |
        |      z  y       |   |
        |      | /        |   |
        |      |/         |   + c2
        |      +-- x      |  /
        |                 | /
        |                 |/
        +-----------------+ c3
      c0=(x1, y1, z1)

      faces: front, back, right, left, up, and down. (borrowed from rubik's cube notation)

    """
    if debug_timing:
        print()
        for i in range(n):
            print(f"cube/sphere {i}:")
            print(f"  world__T__cube:", world__T__cuboids[i])
            print(f"  corners cube:  ", cuboid_corners[i])
            print(f"  sphere center: ", sphere_centers[i])
            print(f"  sphere radius: ", sphere_radii[i])
            print()
        t0 = time()

    # Put everything in cuboid frame
    c0, c1, c2, c3, c4, c5, c6, c7 = CuboidUtils._cuboid_corners_in_world_frame(world__T__cuboids, cuboid_corners)
    G, h = CuboidUtils._get_cuboid_G_h(world__T__cuboids, c0, c1, c2, c3, c4, c5, c6, c7, viz=viz)

    Q = torch.diag_embed(torch.ones(n, 3))
    Q = Q + 5e-4 * torch.eye(3).expand(n, 3, 3)
    p = torch.cat(
        [
            -2 * sphere_centers[:, 0].view(n, 1),
            -2 * sphere_centers[:, 1].view(n, 1),
            -2 * sphere_centers[:, 2].view(n, 1),
        ],
        dim=1,
    ).view(n, 3)
    if debug_timing:
        tsetup = 1000 * (time() - t0)
        tsolve_0 = time()

    # Solve the QP
    if use_qpth:
        import qpth

        e = torch.Tensor()  # Dummy equality constraint
        sol = qpth.qp.QPFunction(verbose=False)(2 * Q, p, G, h, e, e)
        sol = sol.unsqueeze(2)
    else:
        x0 = world__T__cuboids[:, :3, 3].view(n, 3, 1)
        qp = QP(2 * Q, p, G, h, x0)
        sol = qp.solve().unsqueeze(2)

    constraint_violation = G.bmm(sol) - h.view(n, 6, 1)
    assert constraint_violation.max() <= 2e-3, (
        f"constraint violation found\n G * sol - h: {constraint_violation.shape}\nmax violation:"
        f" {constraint_violation.max().item():.16f} / {constraint_violation.max().item()}"
    )

    if debug_timing:
        tsolve = 1000 * (time() - tsolve_0)
        print(f"debug_timing:\n  solve:\t{tsolve:.2f}\n  setup:\t{tsetup:.2f}")

    if debug:
        print()
        for i in range(n):
            print(f"cube/sphere {i}:")
            print(f"  sol: ", sol[i].tolist())
            const = G.matmul(sol[i]) - h[i].view(6, 1)
            print(f"  const: ", const.view(6).tolist())
            print()

    # TODO: add more tests to test_sphere_cuboid()
    dist = torch.norm(sol[:, :, 0] - sphere_centers, dim=1, keepdim=True) - sphere_radii
    if return_sol:
        return dist, sol, G, h
    return dist


def sphere_capsule_distance_batch(
    capsules: torch.Tensor, capsule_poses: torch.Tensor, spheres: torch.Tensor
) -> torch.Tensor:
    """
    Computes the distance between a batch of capsules and a batch of spheres.

    Capsules are defined by two points in local frame, and a radius. The memory layout is
        [nx7]: [x1, y1, z1, x2, y2, z2, r1].

    Local points are at the origin and top of capsule along the +z axis.

    Sphere are [x y z radius]

    Args:
        capsules (torch.Tensor): [n x 7] tensor describing a batch of capsules.
        capsule_poses (torch.Tensor): [n x 4 x 4] tensor describing the pose of the capsules.
        spheres (torch.Tensor): [m x 4] tensor describing a batch of spheres.

    Returns:
        torch.Tensor: [n] tensor containing the distance between capsule i and sphere i for i in [0, n).
    """
    n_batch = capsules.shape[0]
    sphere_radius = spheres[:, 3]
    caps_radius = capsules[:, 6]
    caps_p1 = (capsule_poses[:, :3, :3].bmm(capsules[:, 0:3].unsqueeze(2)).squeeze(2) + capsule_poses[:, :3, 3])[:, 0:3]
    caps_p2 = (capsule_poses[:, :3, :3].bmm(capsules[:, 3:6].unsqueeze(2)).squeeze(2) + capsule_poses[:, :3, 3])[:, 0:3]
    p_spheres = spheres[:, :3]
    v1 = p_spheres - caps_p1
    d = caps_p2 - caps_p1
    t = torch.sum(v1 * d, dim=1) / torch.sum(d * d, dim=1)
    # t=0 -> v_proj = 0
    # t=1 -> v_proj = d
    t = torch.clamp(t, 0.0, 1.0).view(n_batch, 1)
    closest_point = caps_p1 + t * d
    return torch.norm(closest_point - p_spheres, dim=1) - caps_radius - sphere_radius


# ==========

# c1_dist = torch.norm(caps_p1 - spheres[:, 0:3])
# c2_dist = torch.norm(caps_p2 - spheres[:, 0:3])

# print("\nspheres:\n", spheres)
# print("\ncaps_p1:\n", caps_p1)
# print("\ncaps_p2:\n", caps_p2)
# print("\nc1_dist:\n", c1_dist)
# print("\nc2_dist:\n", c2_dist)

# d = caps_p2 - caps_p1
# v1 = spheres[:, :3] - caps_p1
# v2 = torch.sum(v1 * d, dim=1) / torch.norm(d) * d
# # v2 =  torch.dot(v1, d) / torch.norm(d) * d
# perpendicular = v1 - v2
# return torch.min(torch.min(c1_dist, c2_dist), torch.norm(perpendicular, dim=1)) - caps_radius - sphere_radius


""" python jrl/collision_detection.py
"""


class PlottingDemos:
    @staticmethod
    def plot_sphere_cuboid():
        #
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

        # run
        viz = meshcat_utils.init_vis()
        distances, sol, G, h = cuboid_sphere_distance_batch(
            tfs,
            corners,
            sphere_centers,
            sphere_radii,
            debug_timing=True,
            return_sol=True,
            viz=viz,
            debug=True,
            use_qpth=False,
        )

        # visualize
        #
        color_colliding = [1.0, 0.0, 0.0, 0.4]
        color_non_colliding = [0.0, 1.0, 0.0, 0.4]
        sphere_id = meshcat_utils.add_sphere(viz, sphere_centers, sphere_radii)
        sphere_id = meshcat_utils.add_line(viz, sol[0].view(3), sphere_centers.view(3))
        cube_id = meshcat_utils.add_cuboid(viz, tfs, corners)

        if distances.item() < 0:
            print(f"  collision found ({distances[0].item()})")
            meshcat_utils.set_sphere_color(viz, sphere_id, color_colliding)
            meshcat_utils.set_cuboid_color(viz, cube_id, color_colliding)
        else:
            meshcat_utils.set_sphere_color(viz, sphere_id, color_non_colliding)
            meshcat_utils.set_cuboid_color(viz, cube_id, color_non_colliding)

        meshcat_utils.spin(viz)


if __name__ == "__main__":
    PlottingDemos.plot_sphere_cuboid()
