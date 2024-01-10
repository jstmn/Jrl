import sys
import os
from time import sleep
import argparse
from typing import List

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, "..")

import meshcat
import torch
import numpy as np

from jrl.robots import get_robot
from jrl.robot import Robot
import jrl

torch.set_default_device(jrl.config.DEVICE)


class Capsule:
    def __init__(self, p1, p2, r):
        super(Capsule, self).__init__()
        self.length = np.linalg.norm(p2 - p1)
        self.radius = r
        self.T = np.eye(4)
        self.T[:3, 3] = (p1 + p2) / 2
        v = p2 - p1
        v = v / np.linalg.norm(v)
        x, y, z = v
        sign = 1 if z > 0 else -1
        a = -1 / (sign + z)
        b = x * y * a
        t1 = np.array([1 + sign * x * x * a, sign * b, -sign * x])
        t2 = np.array([b, sign + y * y * a, -y])
        self.T[:3, 0] = t1
        self.T[:3, 1] = v  # Meshcat uses y as axis of rotational symmetry
        self.T[:3, 2] = t2


def init_vis(links: List[str], vis_mesh_path: str):
    vis = meshcat.Visualizer()
    Tcube = meshcat.transformations.translation_matrix([0.4, 0.1, 0.5]) @ meshcat.transformations.rotation_matrix(
        0.7, [1, 2, 3]
    )
    cube_lengths = np.array([0.4, 0.5, 0.3])
    vis[f"cuboid"].set_object(
        meshcat.geometry.Box(cube_lengths), meshcat.geometry.MeshToonMaterial(color=0xFF8800, wireframe=True)
    )
    vis[f"cuboid"].set_transform(Tcube)

    for link_i, link in enumerate(links):
        capsule = robot._collision_capsules[link_i, :].cpu().numpy().astype(np.float64)
        p1, p2, capsule_radius = capsule[0:3], capsule[3:6], capsule[6]
        capsule_geom = Capsule(p1, p2, capsule_radius)
        capsule_material = meshcat.geometry.MeshToonMaterial(color=0x8888FF, opacity=0.4)

        vis[f"{robot.name}/{link}/capsule/p1"].set_transform(meshcat.transformations.translation_matrix(p1))
        vis[f"{robot.name}/{link}/capsule/p2"].set_transform(meshcat.transformations.translation_matrix(p2))
        vis[f"{robot.name}/{link}/capsule/cyl"].set_transform(capsule_geom.T)
        vis[f"{robot.name}/{link}/mesh"].set_object(
            meshcat.geometry.DaeMeshGeometry.from_file(f"{vis_mesh_path}/{link}.dae"),
            meshcat.geometry.MeshLambertMaterial(color=0xFFFFFF),
        )
        vis[f"{robot.name}/{link}/capsule/p1"].set_object(meshcat.geometry.Sphere(capsule_radius), capsule_material)
        vis[f"{robot.name}/{link}/capsule/p2"].set_object(meshcat.geometry.Sphere(capsule_radius), capsule_material)
        cyl_geom = meshcat.geometry.Cylinder(capsule_geom.length, capsule_radius)
        vis[f"{robot.name}/{link}/capsule/cyl"].set_object(cyl_geom, capsule_material)

    return vis, Tcube, cube_lengths


def set_config(
    vis: meshcat.Visualizer,
    robot: Robot,
    q: torch.Tensor,
    links: List[str],
    cube_lengths: np.ndarray,
    Tcube: np.ndarray,
):
    n = robot._capsule_idx_to_link_idx.shape[0]
    base_T_links = robot.forward_kinematics_batch(q, return_full_link_fk=True, out_device=q.device, dtype=q.dtype)
    T1s = base_T_links[:, robot._collision_idx0, :, :].reshape(-1, 4, 4)
    T2s = base_T_links[:, robot._collision_idx1, :, :].reshape(-1, 4, 4)
    c1s = robot._collision_capsules[robot._collision_idx0, :].expand(1, -1, -1).reshape(-1, 7)
    c2s = robot._collision_capsules[robot._collision_idx1, :].expand(1, -1, -1).reshape(-1, 7)
    self_dists = jrl.geometry.capsule_capsule_distance_batch(c1s, T1s, c2s, T2s).reshape(1, -1)

    caps = robot._collision_capsules
    Tcaps = base_T_links[:, robot._capsule_idx_to_link_idx, :, :].reshape(-1, 4, 4)
    x, y, z = cube_lengths.astype(np.float32) / 2
    cubes = torch.tensor([[-x, -y, -z, x, y, z]]).expand(n, 6)
    Tcubes = torch.tensor(Tcube, dtype=torch.float32).expand(n, 4, 4)
    env_dists = jrl.geometry.capsule_cuboid_distance_batch(caps, Tcaps, cubes, Tcubes)

    self_colliding = False
    env_colliding = False

    for link_i, link in enumerate(links):
        T = base_T_links[0, link_i, :, :].cpu().numpy().astype(np.float64)
        vis[f"{robot.name}/{link}"].set_transform(T)

        color = [0.0, 1.0, 0.0, 0.4]
        is_self_collide = torch.any(self_dists[0, robot._collision_idx0 == link_i] < 0) or torch.any(
            self_dists[0, robot._collision_idx1 == link_i] < 0
        )
        if is_self_collide:
            color = [1.0, 0.0, 0.0, 0.4]
            self_colliding = True

        is_env_collide = env_dists[link_i, 0] < 0
        if is_env_collide:
            color = [1.0, 0.5, 0.0, 0.4]
            env_colliding = True

        vis[f"{robot.name}/{link}/capsule/p1"].set_property("color", color)
        vis[f"{robot.name}/{link}/capsule/p2"].set_property("color", color)
        vis[f"{robot.name}/{link}/capsule/cyl"].set_property("color", color)

    return self_colliding, env_colliding


def main(robot: Robot):
    links = list(k for k, v in robot._collision_capsules_by_link.items() if v is not None)
    if "fetch" in robot.name:
        vis_mesh_path = "jrl/urdfs/fetch/meshes"
    elif "rizon" in robot.name:
        vis_mesh_path = "jrl/urdfs/rizon4/meshes/visual"
        links.remove("base_link")
        links.insert(0, "link0")
    elif "panda" in robot.name:
        vis_mesh_path = "jrl/urdfs/panda/meshes/visual"
        links = [link.replace("panda_", "") for link in links]
        links.remove("link8")

    vis, Tcube, cube_lengths = init_vis(links, vis_mesh_path)
    q = torch.zeros((1, robot.ndof))

    for t in range(100):
        joint_idx = t % robot.ndof
        l, u = robot.actuated_joints_limits[joint_idx]
        q[0, joint_idx] = (u - l) * torch.rand(1) + l
        q = robot.clamp_to_joint_limits(q)

        is_self_collision, is_env_collision = set_config(vis, robot, q, links, cube_lengths, Tcube)

        sleep(0.1)
        if is_self_collision or is_env_collision:
            sleep(1.0)


""" 
python scripts/visualize_robot_meshcat.py --robot_name panda
python scripts/visualize_robot_meshcat.py --robot_name fetch
python scripts/visualize_robot_meshcat.py --robot_name rizon4

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, help="Example: 'panda', 'fetch', ...")
    args = parser.parse_args()
    robot = get_robot(args.robot_name)
    main(robot)
