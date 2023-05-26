import torch
import functorch
import numpy as np
import stl
import pathlib

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


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


def lm_penalty_optimal_capsule(vertices: torch.Tensor):
    p1_0 = torch.tensor([0.0, 0.0, -10.0])
    p2_0 = torch.tensor([0.0, 0.0, 10.0])
    r_0 = torch.tensor([10])
    x = torch.cat((p1_0, p2_0, r_0), dim=0)

    def fg(x, mu, vertices):
        p1, p2, r = x[None, 0:3], x[None, 3:6], x[6:7]
        dists = point_capsule_distance_batch(
            vertices,
            p1.expand(vertices.shape[0], -1),
            p2.expand(vertices.shape[0], -1),
            r.expand(vertices.shape[0]),
        )
        return torch.cat(
            (
                capsule_volume_batch(p1, p2, r),
                torch.clamp(mu * dists, min=0),
            )
        )

    Jfn = functorch.jacfwd(fg, argnums=0)

    margin = 1e-3
    xtol = 1e-6
    mu = 1.0
    outer_step = 0
    satisfied = False
    while not satisfied:
        inner_step = 0
        converged = False
        while not converged:
            J = Jfn(x, mu, vertices)
            A = J.t() @ J
            lmbd = torch.min(torch.real(torch.linalg.eigvals(A)))
            A += (lmbd + 1e-2) * torch.eye(J.shape[1])
            b = -J.t() @ fg(x, mu, vertices)
            dx = torch.linalg.solve(A, b)
            x = x + dx
            if torch.norm(dx) < xtol:
                converged = True
                print(f"Converged in {inner_step} steps")

            if inner_step > 1000:
                print(f"Did not converge in {inner_step} inner LM steps")
                break

            inner_step += 1

        if torch.all(fg(x, mu, vertices)[1:] <= margin):
            satisfied = True
            print(f"Satisfied in {outer_step} outer steps")

        if outer_step > 100:
            print(f"Did not satisfy in {outer_step} outer steps")
            break

        mu *= 10
        outer_step += 1

    return x[0:3], x[3:6], x[6] + margin


def stl_to_capsule(stl_path: str, outdir):
    mesh = stl.mesh.Mesh.from_file(stl_path)
    vertices = mesh.vectors.reshape(-1, 3)
    vertices = torch.tensor(vertices)

    p1, p2, r = lm_penalty_optimal_capsule(vertices)
    print("p1", p1)
    print("p2", p2)
    print("r", r)

    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
    plot_sphere(axes, p1, r)
    plot_sphere(axes, p2, r)

    scale = mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    linkname = stl_path.stem
    img_path = outdir / f"{linkname}.png"
    print(f"Rendering to {img_path}")
    plt.savefig(img_path)

    txt_path = outdir / f"{linkname}.txt"
    print(f"Saving capsule to {txt_path}")
    with open(txt_path, "w") as f:
        f.write(f"{p1[0]}, {p1[1]}, {p1[2]}, {p2[0]}, {p2[1]}, {p2[2]}, {r}\n")


def main():
    outdir = pathlib.Path("jkinpylib/urdfs/panda/capsules")
    outdir.mkdir(exist_ok=False)
    for stl_path in pathlib.Path("jkinpylib/urdfs/panda/meshes/collision").glob("*.stl"):
        stl_to_capsule(stl_path, outdir)

    outdir = pathlib.Path("jkinpylib/urdfs/fetch/capsules")
    outdir.mkdir(exist_ok=False)
    for stl_path in pathlib.Path("jkinpylib/urdfs/fetch/meshes").glob("*_collision.STL"):
        stl_to_capsule(stl_path, outdir)


if __name__ == "__main__":
    main()
