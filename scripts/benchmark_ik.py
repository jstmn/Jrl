from time import time
from typing import Callable

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from jrl.utils import to_torch, set_seed
from jrl.robots import Panda


def fn_mean_std(fn: Callable, k: int):
    runtimes = []
    for _ in range(k):
        t0 = time()
        fn()
        runtimes.append(1000 * (time() - t0))
    return np.mean(runtimes), np.std(runtimes)


""" 
Example usage:
    uv run python scripts/benchmark_ik.py

Note: see 'batch_ik_covergence_analysis.ipynb' for further analysis.
"""


if __name__ == "__main__":
    set_seed(30)
    assert torch.cuda.is_available()

    robot = Panda()
    k = 5

    df = pd.DataFrame(
        columns=["method", "number of solutions", "total runtime (ms)", "runtime std", "runtime per solution (ms)"]
    )

    for batch_size in [1, 10, 50, 100, 500, 1000, 5000]:
        print(f"Batch size: {batch_size}")

        goalangles, goalposes = robot.sample_joint_angles_and_poses(batch_size)
        x = robot.sample_joint_angles(batch_size)

        goalposes_cpu = to_torch(goalposes.copy()).cpu()
        goalposes_cuda = to_torch(goalposes.copy()).cuda()

        x_pt_cpu = to_torch(x.copy()).cpu()
        x_pt_cuda = to_torch(x.copy()).cuda()

        methods = {
            "Levenberg-Marquardt": lambda: robot.inverse_kinematics_step_levenburg_marquardt(goalposes_cuda, x_pt_cuda),
            "Levenberg-Marquardt Cholesky": lambda: robot.inverse_kinematics_step_levenburg_marquardt_cholesky(
                goalposes_cuda, x_pt_cuda
            ),
            "Jacobian Psuedo-Inverse": lambda: robot.inverse_kinematics_step_jacobian_pinv(goalposes_cuda, x_pt_cuda),
            # "torch.AutoDiff": lambda: robot.inverse_kinematics_autodiff_single_step_batch_pt(goalposes_cuda, x_pt_cuda),
        }
        for name, method in methods.items():
            mean_runtime_ms, std_runtime = fn_mean_std(method, k)
            new_row = [name, batch_size, mean_runtime_ms, std_runtime, mean_runtime_ms / batch_size]
            df.loc[len(df)] = new_row

    df = df.sort_values(by=["method", "number of solutions"])

    print(df)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for method_name in methods.keys():
        df_method = df[df["method"] == method_name]
        n_solutions = df_method["number of solutions"]
        total_runtime_ms = df_method["total runtime (ms)"]
        std = df_method["runtime std"]
        label = method_name
        p = ax.plot(n_solutions, total_runtime_ms, label=label)
        ax.fill_between(n_solutions, total_runtime_ms - std, total_runtime_ms + std, alpha=0.2, color=p[0].get_color())

    ax.set_title("IK Update Method Runtime Comparison")
    ax.set_xlabel("Number of solutions")
    ax.set_ylabel("Runtime [ms]")
    ax.grid(True, which="both", ls="--", c="gray", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", ls="--", c="gray", alpha=0.15)
    ax.legend()
    fig.savefig("media/ik_runtime_benchmark.png", bbox_inches="tight")
