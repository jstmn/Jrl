import torch
import numpy as np
import stl
import pathlib
import meshcat
import argparse

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

from jrl.robots import ALL_ROBOT_NAMES


def capsule_volume_batch(c1: torch.Tensor, c2: torch.Tensor, r: torch.Tensor):
    """Compute the volume of a capsule given its end points and radius.

    Args:
        c1 [batch x 3]: First end point.
        c2 [batch x 3]: Second end point.
        r [batch]: Radius of the capsule.

    Returns:
        [batch]: Volume of the capsule.
    """

    h = torch.norm(c2 - c1, dim=1)
    return np.pi * h * (r**2) + (4 / 3) * np.pi * (r**3)


def point_capsule_distance_batch(p: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, r: torch.Tensor):
    """Compute the distance between a point and a capsule given its end points and radius.

    Args:
        p [batch x 3]: Point.
        c1 [batch x 3]: First end point.
        c2 [batch x 3]: Second end point.
        r [batch]: Radius of the capsule.

    Returns:
        [batch]: Distance between the point and the capsule.
    """

    pc1 = p - c1
    c2c1 = c2 - c1
    h = torch.clamp((pc1 * c2c1).sum(dim=1) / (c2c1 * c2c1).sum(dim=1), 0, 1)
    return torch.norm(pc1 - h.unsqueeze(1) * c2c1, dim=1) - r


def plot_sphere(ax, center, radius):
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_wireframe(x, y, z, color="r")


def random_choice_optimal_capsule(vertices: torch.Tensor):
    nsamples = 100000
    p1s = 0.05 * torch.randn(nsamples, 3)
    p2s = 0.05 * torch.randn(nsamples, 3)
    rs = 0.05 * torch.abs(torch.randn(nsamples))

    dists = point_capsule_distance_batch(
        vertices.unsqueeze(0).expand(nsamples, -1, -1).reshape(-1, 3),
        p1s.unsqueeze(1).expand(-1, vertices.shape[0], -1).reshape(-1, 3),
        p2s.unsqueeze(1).expand(-1, vertices.shape[0], -1).reshape(-1, 3),
        rs.unsqueeze(1).expand(-1, vertices.shape[0]).reshape(-1),
    ).reshape(nsamples, vertices.shape[0])
    maxdists, _ = torch.max(dists, dim=1)
    mask = maxdists < 0
    _, i = torch.min(capsule_volume_batch(p1s[mask], p2s[mask], rs[mask]), dim=0)
    p1, p2, r = p1s[mask][i], p2s[mask][i], rs[mask][i]

    return p1, p2, r


def lm_penalty_optimal_capsule(vertices: torch.Tensor, nruns=5, vis=None):
    best_p1, best_p2, best_r = None, None, None
    best_cost = np.inf
    for i in range(nruns):
        try:
            # p1_0 = torch.tensor([0.0, 0.0, -10.0])
            p1_0 = torch.randn(3)
            p1_0 = 0.2 * p1_0 / torch.norm(p1_0)
            # p2_0 = torch.tensor([0.0, 0.0, 10.0])
            # p2_0 = 0.5 * torch.randn(3)
            p2_0 = -p1_0
            # r_0 = torch.tensor([10])
            r_0 = torch.abs(torch.randn(1))
            x = torch.cat((p1_0, p2_0, r_0), dim=0)

            def fg(x, mu, vertices):
                p1, p2, r = x[None, 0:3], x[None, 3:6], x[6:7]
                dists = point_capsule_distance_batch(
                    vertices,
                    p1.expand(vertices.shape[0], -1),
                    p2.expand(vertices.shape[0], -1),
                    r.expand(vertices.shape[0]),
                )
                return torch.cat((
                    capsule_volume_batch(p1, p2, r),
                    torch.clamp(mu * dists, min=0),
                ))

            Jfn = torch.func.jacfwd(fg, argnums=0)

            margin = 1e-3
            xtol = 1e-6
            mu = 0.1
            outer_step = 0
            satisfied = False
            while not satisfied:
                inner_step = 0
                converged = False
                while not converged:
                    y = x.cpu()
                    p1vis = np.array([y[0], y[1], y[2]], dtype=np.float64)
                    p2vis = np.array([y[3], y[4], y[5]], dtype=np.float64)
                    rvis = y[6].item()
                    h = np.linalg.norm(p2vis - p1vis)
                    capsule_material = meshcat.geometry.MeshToonMaterial(color=0x8888FF, opacity=0.4)
                    if vis is not None:
                        vis["p1"].set_object(meshcat.geometry.Sphere(rvis), capsule_material)
                        vis["p1"].set_transform(meshcat.transformations.translation_matrix(p1vis))
                        vis["p2"].set_object(meshcat.geometry.Sphere(rvis), capsule_material)
                        vis["p2"].set_transform(meshcat.transformations.translation_matrix(p2vis))
                        vis["cyl"].set_object(meshcat.geometry.Cylinder(h, rvis), capsule_material)
                    T = np.eye(4)
                    T[:3, 3] = (p1vis + p2vis) / 2
                    v = p2vis - p1vis
                    if np.linalg.norm(v) > 1e-6:
                        v = v / np.linalg.norm(v)
                    else:
                        v = np.array([0, 0, 1])
                    vx, vy, vz = v
                    sign = 1 if vz > 0 else -1
                    a = -1 / (sign + vz)
                    b = vx * vy * a
                    t1 = np.array([1 + sign * vx * vx * a, sign * b, -sign * vx])
                    t2 = np.array([b, sign + vy * vy * a, -vy])
                    T[:3, 0] = t1
                    T[:3, 1] = v  # Meshcat uses y as axis of rotational symmetry
                    T[:3, 2] = t2
                    if vis is not None:
                        vis["cyl"].set_transform(T)

                    J = Jfn(x, mu, vertices)
                    A = J.t() @ J
                    # lmbd = max(0, torch.min(torch.real(torch.linalg.eigvals(A))).item())
                    A += (1e-6) * torch.eye(J.shape[1])
                    b = -J.t() @ fg(x, mu, vertices)
                    dx = torch.linalg.solve(A, b)
                    x = x + 1e-2 / mu * dx
                    if torch.norm(dx) < xtol:
                        converged = True
                        print(f"Converged in {inner_step} steps")

                    if inner_step > 5000:
                        print(f"Did not converge in {inner_step} inner LM steps")
                        break

                    inner_step += 1
                    mins = torch.min(vertices, dim=0)[0]
                    maxs = torch.max(vertices, dim=0)[0]
                    x = torch.clamp(
                        x,
                        min=torch.cat([mins, mins, torch.tensor([0.01])]),
                        max=torch.cat([maxs, maxs, torch.tensor([1])]),
                    )

                if torch.all(fg(x, 1, vertices)[1:] <= margin):
                    satisfied = True
                    print(f"Satisfied in {outer_step} outer steps")

                if outer_step > 100:
                    print(f"Did not satisfy in {outer_step} outer steps")
                    break

                mu *= 2
                outer_step += 1

            if satisfied:
                p1, p2, r = x[0:3], x[3:6], x[6]
                cost = capsule_volume_batch(p1.reshape(-1, 3), p2.reshape(-1, 3), r.reshape(-1))
                if cost < best_cost:
                    best_p1, best_p2, best_r = p1, p2, r
                    best_cost = cost

        except Exception as e:
            print(e)

    return best_p1, best_p2, best_r + margin


def stl_to_capsule(stl_path: str, outdir: pathlib.PosixPath, vis=None):

    linkname = stl_path.stem
    txt_path = outdir / f"{linkname}.txt"
    if txt_path.exists():
        print(f"\nStl '{stl_path}' has been converted already - skipping")
        return

    print(f"\nstl_to_capsule() | Approximating {stl_path}")
    stl_mesh_geom = meshcat.geometry.StlMeshGeometry.from_file(stl_path)
    if vis is not None:
        vis["mesh"].set_object(stl_mesh_geom)
    mesh = stl.mesh.Mesh.from_file(stl_path)
    vertices = mesh.vectors.reshape(-1, 3)
    vertices = torch.tensor(vertices)

    p1, p2, r = lm_penalty_optimal_capsule(vertices, vis=vis)
    print("p1", p1)
    print("p2", p2)
    print("r", r)

    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
    plot_sphere(axes, p1.cpu().numpy(), r.cpu().numpy())
    plot_sphere(axes, p2.cpu().numpy(), r.cpu().numpy())

    scale = mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    img_path = outdir / f"{linkname}.png"
    print(f"Rendering to {img_path}")
    plt.savefig(img_path)

    print(f"Saving capsule to {txt_path}")
    with open(txt_path, "w") as f:
        f.write(f"{p1[0]}, {p1[1]}, {p1[2]}, {p2[0]}, {p2[1]}, {p2[2]}, {r}\n")

    print(f"Done with '{stl_path}'")


"""
python scripts/calculate_capsule_approximation.py --visualize --robot_name=iiwa14
python scripts/calculate_capsule_approximation.py --visualize --robot_name=iiwa7
"""


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--visualize", action="store_true")
    argparser.add_argument("--robot_name", type=str, required=True)
    args = argparser.parse_args()

    assert args.robot_name in ALL_ROBOT_NAMES

    vis = None
    if args.visualize:
        vis = meshcat.Visualizer()
        vis.open()

    outdir = pathlib.Path(f"jrl/urdfs/{args.robot_name}/capsules")
    outdir.mkdir(exist_ok=True)
    for stl_path in pathlib.Path(f"jrl/urdfs/{args.robot_name}/meshes/collision").glob("*.stl"):
        stl_to_capsule(stl_path, outdir, vis)
