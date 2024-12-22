from typing import Tuple, Optional
from time import sleep

import colorsys
import uuid
import meshcat as mc
import numpy as np
import torch


"""
A cylinder of the given height and radius. By Three.js convention, the axis of rotational symmetry is aligned with the 
y-axis.

Borrowed from the 'Cylinder' class in geometry.py from mc
"""


class OpenEndedCylinder(mc.geometry.Geometry):
    field = "geometries"

    def __init__(self, height, radius=1.0, radiusTop=None, radiusBottom=None):
        super(mc.geometry.Geometry, self).__init__()
        if radiusTop is not None and radiusBottom is not None:
            self.radiusTop = radiusTop
            self.radiusBottom = radiusBottom
        else:
            self.radiusTop = radius
            self.radiusBottom = radius
        self.height = height
        self.radialSegments = 50
        # self.radialSegments = 100
        # self.radialSegments = 75

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "CylinderGeometry",
            "radiusTop": self.radiusTop,
            "radiusBottom": self.radiusBottom,
            "height": self.height,
            "radialSegments": self.radialSegments,
            "openEnded": True,
        }


"""
A cylinder of the given height and radius. By Three.js convention, the axis of
rotational symmetry is aligned with the y-axis.
"""


class JmCapsule(mc.geometry.Geometry):
    def __init__(self, length, radius):
        super(JmCapsule, self).__init__()
        self.length = length
        self.radius = radius
        self.capSegments = 4
        self.radialSegments = 4

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "radius": self.radius,  #  — Radius of the capsule. Optional; defaults to 1.
            "length": self.length,  #  — Length of the middle section. Optional; defaults to 1.
            "capSegments": (
                self.capSegments
            ),  #  — Number of curve segments used to build the caps. Optional; defaults to 4.
            "radialSegments": self.radialSegments,
            "type": "CapsuleGeometry",
        }


num_cuboids = 0
num_capsules = 0
num_pointclouds = 0
# DEFAULT_MATERIAL = mc.geometry.MeshToonMaterial(color=0xFF8800, wireframe=True)
DEFAULT_MATERIAL = mc.geometry.MeshPhongMaterial(color=0xFF8800, wireframe=True)
# DEFAULT_MATERIAL = mc.geometry.MeshLambertMaterial(color=0xFF8800, wireframe=True)
# DEFAULT_MATERIAL = mc.geometry.MeshLambertMaterial(color=0xFF8800, wireframe=False)
DEFAULT_POINT_MATERIAL = mc.geometry.PointsMaterial()


def k_evenly_spaced_colors(k: int) -> Tuple[float, float, float]:
    """Returns k evenly spaced colors in RGB space."""
    colors = []
    for i in range(k):
        hue = i / k
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        colors.append((r, g, b))
    return colors


def add_cuboid(vis: mc.Visualizer, world__T__cuboid: torch.Tensor, corners: torch.Tensor):
    """_summary_

    corners:
    width — Width; that is, the length of the edges parallel to the X axis. Optional; defaults to 1.
    height — Height; that is, the length of the edges parallel to the Y axis. Optional; defaults to 1.
    depth — Depth; that is, the length of the edges parallel to the Z axis. Optional; defaults to 1.

    Args:
        vis (mc.Visualizer): _description_
        world__T__cuboid (torch.Tensor): _description_
        corners (torch.Tensor): _description_
    """
    assert corners.numel() == 6
    assert world__T__cuboid.numel() == 16
    assert (corners.view(6)[0:3].abs() - corners.view(6)[3:].abs()).abs().max() < 1e-6, f"cuboid must be symmetrical"
    global num_cuboids
    corners = corners.view(6).cpu().numpy().astype(np.float64)
    side_lengths = np.array(corners[3:] - corners[0:3])
    tf = world__T__cuboid.view(4, 4).cpu().numpy().astype(np.float64)  # note: needs to be np.float64
    box = mc.geometry.Box(side_lengths)
    vis[f"cuboid_{num_cuboids}"].set_object(box, DEFAULT_MATERIAL)
    vis[f"cuboid_{num_cuboids}"].set_transform(tf)

    # add sphere to viz center
    # sphere = mc.geometry.Sphere(0.025)
    # vis[f"cuboid_{num_cuboids}_sph"].set_object(sphere, DEFAULT_MATERIAL)
    # vis[f"cuboid_{num_cuboids}_sph"].set_transform(tf)
    num_cuboids += 1
    return num_cuboids - 1


def add_capsule(vis: mc.Visualizer, world__T__capsule: torch.Tensor, cap_pose: torch.Tensor):
    """Add a capsule to the scene.

    NOTE: three.js defines cylinders with the axis of symmetry along the y-axis for some ungodly reason. Therefore, we
    rotate the cylinder in the +90 direction about the x-axis so that the 'length' direction corresponds with the z-axis

    Args:
        vis (mc.Visualizer): _description_
        world__T__capsule (torch.Tensor): _description_
        cap_pose (torch.Tensor): [ 7 ] vector with format [x1, y1, z1, x2, y2, z2, r1].
    """
    assert world__T__capsule.numel() == 16
    assert cap_pose.numel() == 7
    assert cap_pose[0:2].abs().max() < 1e-6, f"x, y components must be 0"
    assert cap_pose[3:5].abs().max() < 1e-6, f"x, y components must be 0"
    global num_capsules
    world__T__capsule = world__T__capsule.view(4, 4).cpu().numpy().astype(np.float64)  # note: needs to be np.float64
    cap_pose = cap_pose.cpu().numpy().astype(np.float64)
    cylinder_height = np.linalg.norm(cap_pose[0:3] - cap_pose[3:6]).item()
    capsule_radius = cap_pose[6]

    cylinder = OpenEndedCylinder(cylinder_height, radius=capsule_radius)
    sphere_low = mc.geometry.Sphere(capsule_radius)
    sphere_high = mc.geometry.Sphere(capsule_radius)
    vis[f"capsule_{num_capsules}_cyl"].set_object(cylinder, DEFAULT_MATERIAL)
    vis[f"capsule_{num_capsules}_sph_l"].set_object(sphere_low, DEFAULT_MATERIAL)
    vis[f"capsule_{num_capsules}_sph_h"].set_object(sphere_high, DEFAULT_MATERIAL)

    world__T__capsule__caps_l = np.eye(4)
    world__T__capsule__caps_h = np.eye(4)
    world__T__capsule__caps_l[0:3, 3] = world__T__capsule[0:3, 3] + world__T__capsule[0:3, 0:3] @ cap_pose[0:3]
    world__T__capsule__caps_h[0:3, 3] = world__T__capsule[0:3, 3] + world__T__capsule[0:3, 0:3] @ cap_pose[3:6]
    world__T__capsule[0:3, 0:3] = (
        world__T__capsule[0:3, 0:3] @ mc.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]
    )

    vis[f"capsule_{num_capsules}_cyl"].set_transform(world__T__capsule)
    vis[f"capsule_{num_capsules}_sph_l"].set_transform(world__T__capsule__caps_l)
    vis[f"capsule_{num_capsules}_sph_h"].set_transform(world__T__capsule__caps_h)
    num_capsules += 1
    return num_capsules - 1


def add_pointcloud(vis: mc.Visualizer, points: torch.Tensor, color: Optional[Tuple] = (0.5, 1.0, 0.5)):

    points = points.T
    n = points.shape[1]
    points = points.cpu().numpy().astype(np.float64)
    color_repeated = np.array([[color[0]] * n, [color[1]] * n, [color[2]] * n])
    assert (
        color_repeated.shape == points.shape
    ), f"color_repeated.shape: {color_repeated.shape}, points.shape: {points.shape}"
    global num_pointclouds

    point_cloud = mc.geometry.Points(
        mc.geometry.PointsGeometry(points, color=color_repeated), mc.geometry.PointsMaterial(size=0.075)
    )
    vis[f"pcd_{num_pointclouds}"].set_object(point_cloud)
    num_pointclouds += 1
    return num_pointclouds - 1


num_spheres = 0


def add_sphere(
    vis: mc.Visualizer, center: torch.Tensor, radius: torch.Tensor, color: Optional[Tuple] = (0.5, 1.0, 0.5)
):
    assert center.numel() == 3
    global num_spheres
    if isinstance(radius, torch.Tensor):
        assert radius.numel() == 1
        radius = radius.item()

    tf = mc.transformations.identity_matrix()
    tf[0:3, 3] = center.cpu().numpy().astype(np.float64)
    sphere = mc.geometry.Sphere(radius)
    vis[f"sphere_{num_spheres}"].set_object(sphere, DEFAULT_MATERIAL)
    vis[f"sphere_{num_spheres}"].set_transform(tf)
    if color is not None:
        vis[f"sphere_{num_spheres}"].set_property("color", color)
    num_spheres += 1
    return num_spheres - 1


num_lines = 0


def add_line(vis: mc.Visualizer, p1: torch.Tensor, p2: torch.Tensor, color: Optional[Tuple] = (0.0, 0.0, 0.2)):
    assert p1.numel() == 3
    assert p2.numel() == 3
    assert p1.ndim == 1
    assert p2.ndim == 1
    global num_lines
    """Plot a plane characterized by the equation a^T x = b"""
    points = []
    n = int(50 * torch.norm(p1 - p2).item())
    xs = torch.linspace(p1[0], p2[0], n).view(n, 1)
    ys = torch.linspace(p1[1], p2[1], n).view(n, 1)
    zs = torch.linspace(p1[2], p2[2], n).view(n, 1)
    points = torch.cat([xs, ys, zs], dim=1).T
    points = points.cpu().numpy().astype(np.float64)
    color_repeated = np.array([[color[0]] * n, [color[1]] * n, [color[2]] * n]).astype(np.float64)
    assert (
        color_repeated.shape == points.shape
    ), f"color_repeated.shape: {color_repeated.shape}, points.shape: {points.shape}"
    point_cloud = mc.geometry.Points(
        mc.geometry.PointsGeometry(points, color=color_repeated), mc.geometry.PointsMaterial(size=0.02)
    )
    vis[f"line_{num_lines}"].set_object(point_cloud)
    num_lines += 1
    return num_lines - 1


num_planes = 0


# TODO: replace with meshcat.geometry.Box
def add_plane(
    vis: mc.Visualizer,
    a: torch.Tensor,
    b: float,
    center_point: Optional[torch.Tensor] = None,
    color: Optional[Tuple] = (0.5, 1.0, 0.5),
):
    """Plot a plane characterized by the equation a^T x = b

    If center_point is provided, that is the center of sampled (x, y, z) points. Otherwise, the center is assumed to be
    the intersection point between the vector pointing from the origin to the plane.
    """
    assert isinstance(a, torch.Tensor)
    assert a.numel() == 3
    if isinstance(b, torch.Tensor):
        assert b.numel() == 1
        b = b.item()
    assert isinstance(b, float)
    a0, a1, a2 = a.view(3).tolist()
    points = []

    #
    if center_point is not None:
        assert center_point.numel() == 3
        assert isinstance(center_point, torch.Tensor)
    else:
        # see https://www.desmos.com/3d/wxyl7dfkkq
        center_point = (b / (a0**2 + a1**2 + a2**2)) * a

    # sample points around the center point
    sample_width = 1.5
    if abs(a2) > 1e-4:
        n_samples = 30
        xs = (
            torch.linspace(center_point[0] - sample_width, center_point[0] + sample_width, n_samples)
            .view(n_samples, 1)
            .repeat_interleave(n_samples, 0)
        )
        ys = (
            torch.linspace(center_point[1] - sample_width, center_point[1] + sample_width, n_samples)
            .view(n_samples, 1)
            .repeat(n_samples, 1)
        )
        zs = (b - a0 * xs - a1 * ys) / a2
        points = torch.cat([xs, ys, zs], dim=1)
        points = points[torch.norm(points - center_point, dim=1) < 1.0]
    else:
        assert False, f"not implemented"

    assert len(points) > 0, f"no points found for plane"
    points = torch.tensor(points).T
    n = points.shape[1]
    points = points.cpu().numpy().astype(np.float64)
    color_repeated = np.array([[color[0]] * n, [color[1]] * n, [color[2]] * n])
    assert (
        color_repeated.shape == points.shape
    ), f"color_repeated.shape: {color_repeated.shape}, points.shape: {points.shape}"
    global num_planes

    point_cloud = mc.geometry.Points(
        mc.geometry.PointsGeometry(points, color=color_repeated), mc.geometry.PointsMaterial(size=0.02)
    )
    vis[f"plane_{num_planes}"].set_object(point_cloud)
    num_planes += 1
    return num_planes - 1


def set_cuboid_color(vis, cube_id: int, color: Tuple[float, float, float, float]):
    vis[f"cuboid_{cube_id}"].set_property("color", color)


def set_sphere_color(vis, id_: int, color: Tuple[float, float, float, float]):
    vis[f"sphere_{id_}"].set_property("color", color)


def set_capsule_color(vis, caps_id, color: Tuple[float, float, float, float]):
    vis[f"capsule_{caps_id}_cyl"].set_property("color", color)  # "0xff0000")
    vis[f"capsule_{caps_id}_sph_l"].set_property("color", color)  # "0xff0000")
    vis[f"capsule_{caps_id}_sph_h"].set_property("color", color)  # "0xff0000")


def spin(vis: mc.Visualizer):
    vis.open()
    while True:
        sleep(0.1)


def init_vis() -> mc.Visualizer:
    vis = mc.Visualizer()
    # vis["/Background"].set_property("visible", False)
    # vis["/Background"].set_property("top_color", [1.0, 1.0, 1.0])
    # vis["/Background"].set_property("bottom_color", [0.9, 0.9, 0.9])
    return vis
