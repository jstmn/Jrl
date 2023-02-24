import argparse
from time import time
from typing import Callable

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from jkinpylib.utils import to_torch, set_seed
from jkinpylib.robots import Panda
from jkinpylib.conversions import geodesic_distance_between_quaternions


def fn_mean_std(fn: Callable, k: int):
    runtimes = []
    for _ in range(k):
        t0 = time()
        fn()
        runtimes.append(1000 * (time() - t0))
    return np.mean(runtimes), np.std(runtimes)


""" Example 

python scripts/benchmark_fk.py


"""


if __name__ == "__main__":
    set_seed(30)
    assert torch.cuda.is_available()

    robot = Panda()
    k = 5
    df = pd.DataFrame(
        columns=["method", "number of solutions", "total runtime (ms)", "runtime std", "runtime per solution (ms)"]
    )

    for batch_size in [1, 5, 10, 50, 100, 500, 1000, 5000]:
        print(f"Batch size: {batch_size}")

        goalangles, goalposes = robot.sample_joint_angles_and_poses(batch_size)
        x = robot.sample_joint_angles(batch_size)

        goalposes_cpu = to_torch(goalposes.copy()).cpu()
        goalposes_cuda = to_torch(goalposes.copy()).cuda()

        x_pt_cpu = to_torch(x.copy()).cpu()
        x_pt_cuda = to_torch(x.copy()).cuda()

        methods = {
            # "Klampt invJac cpu": lambda: robot.inverse_kinematics_single_step_batch_pt(goalposes_cpu, x_pt_cpu),
            "Klampt invJac cuda": lambda: robot.inverse_kinematics_single_step_batch_pt(goalposes_cuda, x_pt_cuda),
            "AutoDiff cuda": lambda: robot.inverse_kinematics_autodiff_single_step_batch_pt(goalposes_cuda, x_pt_cuda),
        }
        for name, method in methods.items():
            mean_runtime_ms, std_runtime = fn_mean_std(method, k)
            new_row = [name, batch_size, mean_runtime_ms, std_runtime, mean_runtime_ms / batch_size]
            df.loc[len(df)] = new_row

        if batch_size == 1:
            loss_histories = {}
            for name, method in methods.items():
                goalangles, goalposes = robot.sample_joint_angles_and_poses(batch_size)
                goalposes_cuda = to_torch(goalposes.copy()).cuda()
                x_pt_cuda = to_torch(x.copy()).cuda()
                x_pt_cuda = x_pt_cuda + torch.randn_like(x_pt_cuda) / 10

                loss = np.inf
                loss_history = []
                counter = 0
                while loss > 0.01:
                    x_pt_cuda = robot.inverse_kinematics_autodiff_single_step_batch_pt(goalposes_cuda, x_pt_cuda)

                    current_poses = robot.forward_kinematics_batch(
                        x_pt_cuda, out_device=x_pt_cuda.device, dtype=x_pt_cuda.dtype
                    )

                    t_err = goalposes_cuda[:, 0:3] - current_poses[:, 0:3]
                    R_err = geodesic_distance_between_quaternions(goalposes_cuda[:, 3:7], current_poses[:, 3:7])
                    loss = torch.sum(t_err**2) + torch.sum(R_err**2)
                    loss = torch.norm(t_err, dim=1)
                    loss_history.append(loss.item())

                    counter += 1
                loss_histories[name] = loss_history
                print(f"{name} took {counter} steps to converge")

    df = df.sort_values(by=["method", "number of solutions"])

    print(df)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for method_name in df["method"]:
        df_method = df[df["method"] == method_name]
        n_solutions = df_method["number of solutions"]
        total_runtime_ms = df_method["total runtime (ms)"]
        std = df_method["runtime std"]
        ax.plot(n_solutions, total_runtime_ms, label=method_name)
        ax.fill_between(n_solutions, total_runtime_ms - std, total_runtime_ms + std, alpha=0.2)

    ax.set_xlabel("Number of solutions")
    ax.set_ylabel("Total runtime (ms)")
    ax.legend()
    fig.savefig("scripts/batch_ik_runtime.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for name, loss_history in loss_histories.items():
        ax.plot(loss_history, label=name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("sum square error")
    ax.legend()
    fig.savefig("scripts/batch_ik_convergence.pdf", bbox_inches="tight")
