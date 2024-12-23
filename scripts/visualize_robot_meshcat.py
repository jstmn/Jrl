import sys
import os
from time import sleep
import argparse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, "..")

import meshcat
import torch
import numpy as np

from jrl.meshcat_utils import init_vis, add_cuboid, add_robot, spin, set_robot_configuration
from jrl.robots import get_robot, Panda, PandaTemp
from jrl.robot import Robot
import jrl

torch.set_default_device(jrl.config.DEVICE)




def main(vis: meshcat.Visualizer, robot: Robot):
    # links = list(k for k, v in robot._collision_capsules_by_link.items() if v is not None)

    # Add cuboid
    Tcube = meshcat.transformations.translation_matrix([0.4, 0.1, 0.5]) @ meshcat.transformations.rotation_matrix(
        0.7, [1, 2, 3]
    )
    cube_corners = 0.5*torch.tensor([-0.4, -0.5, -0.3, 0.4, 0.5, 0.3])
    # cube_lengths = torch.tensor([0.4, 0.5, 0.3])
    add_cuboid(vis, Tcube, cube_corners)

    link_data = {}
    link_colors = {}

    if "fetch" in robot.name:
        vis_mesh_path = "jrl/urdfs/fetch/meshes"
        raise NotImplementedError()

    elif "rizon" in robot.name:
        vis_mesh_path = "jrl/urdfs/rizon4/meshes/visual"
        links.remove("base_link")
        links.insert(0, "link0")
        raise NotImplementedError()

    elif robot.name == Panda.name:
        raise NotImplementedError()
        vis_mesh_path = "jrl/urdfs/panda/meshes/visual"
        links = [link.replace("panda_", "") for link in links]
        links.remove("link8")

    elif robot.name == PandaTemp.name:
        link_data = {
            link_name: (f"jrl/urdfs/panda/meshes/visual/{link_name.replace('panda_', '')}.dae", f"jrl/urdfs/panda/meshes/collision/{link_name.replace('panda_', '')}.stl") for link_name in robot._collision_capsules_by_link
        }
        del link_data["panda_link8"] # this is a virtual link, doesn't exist physically
        link_data["panda_leftfinger"] = (f"jrl/urdfs/panda/meshes/visual/finger.dae", f"jrl/urdfs/panda/meshes/collision/finger.stl")
        link_colors["panda_leftfinger"] = 0xff0000

    elif "iiwa14" in robot.name:
        raise NotImplementedError()
        vis_mesh_path = "jrl/urdfs/iiwa14/meshes/visual"
        mesh_format = "stl"
        links.remove("world")
        links.remove("link_ee")
        links.remove("link_ee_kuka")
        links.remove("link_ee_kuka_mft_pneum")
        assert False, "stls have +y and +z flipped for iiwa14, need to fix that"

    robot_id = add_robot(vis, robot, link_data, link_colors=link_colors)

    q = torch.zeros((1, robot.ndof))
    for t in range(100):
        joint_idx = t % robot.ndof
        l, u = robot.actuated_joints_limits[joint_idx]
        q[0, joint_idx] = (u - l) * torch.rand(1) + l
        q = robot.clamp_to_joint_limits(q)

        set_robot_configuration(vis, robot_id, robot, q, link_data)


        # 

        # T1s = base_T_links[:, robot._collision_idx0, :, :].reshape(-1, 4, 4)
        # T2s = base_T_links[:, robot._collision_idx1, :, :].reshape(-1, 4, 4)
        # c1s = robot._collision_capsules[robot._collision_idx0, :].expand(1, -1, -1).reshape(-1, 7)
        # c2s = robot._collision_capsules[robot._collision_idx1, :].expand(1, -1, -1).reshape(-1, 7)
        # self_dists = capsule_capsule_distance_batch(c1s, T1s, c2s, T2s).reshape(1, -1)

        # base_T_links = robot.forward_kinematics(q, return_full_link_fk=True, out_device=q.device, dtype=q.dtype)
        # caps = robot._collision_capsules
        # Tcaps = base_T_links[:, robot._capsule_idx_to_link_idx, :, :].reshape(-1, 4, 4)
        # x, y, z = cube_lengths.astype(np.float32) / 2
        # cubes = torch.tensor([[-x, -y, -z, x, y, z]]).expand(n, 6)
        # Tcubes = torch.tensor(Tcube, dtype=torch.float32).expand(n, 4, 4)
        # env_dists = capsule_cuboid_distance_batch(caps, Tcaps, cubes, Tcubes)


        sleep(5)
        # color = [0.0, 1.0, 0.0, 0.4]
        # is_self_collide = torch.any(self_dists[0, robot._collision_idx0 == link_idx] < 0) or torch.any(
        #     self_dists[0, robot._collision_idx1 == link_idx] < 0
        # )
        # if is_self_collide:
        #     color = [1.0, 0.0, 0.0, 0.4]
        #     self_colliding = True

        # is_env_collide = env_dists[link_idx, 0] < 0
        # if is_env_collide:
        #     color = [1.0, 0.5, 0.0, 0.4]
        #     env_colliding = True

        # vis[f"{robot.name}_{robot_id}/{link_name}/capsule/p1"].set_property("color", color)
        # vis[f"{robot.name}_{robot_id}/{link_name}/capsule/p2"].set_property("color", color)
        # vis[f"{robot.name}_{robot_id}/{link_name}/capsule/cyl"].set_property("color", color)


        # sleep(0.1)
        # if is_self_collision or is_env_collision:
        #     sleep(150.0)


""" 
python scripts/visualize_robot_meshcat.py --robot_name panda
python scripts/visualize_robot_meshcat.py --robot_name panda_temp
python scripts/visualize_robot_meshcat.py --robot_name fetch
python scripts/visualize_robot_meshcat.py --robot_name rizon4
python scripts/visualize_robot_meshcat.py --robot_name iiwa14

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, help="Example: 'panda', 'fetch', ...")
    args = parser.parse_args()
    robot = get_robot(args.robot_name)
    vis = init_vis()
    main(vis, robot)
