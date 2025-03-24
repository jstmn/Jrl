import argparse
from time import time
from typing import Callable

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from jrl.utils import to_torch
from jrl.robots import Panda


def fn_mean_std(fn: Callable, k: int):
    runtimes = []
    for _ in range(k):
        t0 = time()
        fn()
        runtimes.append(1000 * (time() - t0))
    return np.mean(runtimes), np.std(runtimes)


""" Example 

uv run python scripts/benchmark_fk.py


"""


if __name__ == "__main__":
    assert torch.cuda.is_available()

    robot = Panda()
    k = 5
    df = pd.DataFrame(
        columns=["method", "number of solutions", "total runtime (ms)", "runtime std", "runtime per solution (ms)"]
    )

    method_names = ["batch_fk - cpu", "batch_fk - cuda", "klampt"]

    for batch_size in [1, 5, 10, 50, 100, 500, 1000, 5000]:
        print(f"Batch size: {batch_size}")

        x = robot.sample_joint_angles(batch_size)
        x_pt_cpu = to_torch(x.copy()).cpu()
        x_pt_cuda = to_torch(x.copy()).cuda()

        lambdas = [
            lambda: robot.forward_kinematics(x_pt_cpu, out_device="cpu"),
            lambda: robot.forward_kinematics(x_pt_cuda, out_device="cuda"),
            lambda: robot.forward_kinematics_klampt(x),
        ]
        for lambda_, method_name in zip(lambdas, method_names):
            mean_runtime_ms, std_runtime = fn_mean_std(lambda_, k)
            new_row = [method_name, batch_size, mean_runtime_ms, std_runtime, mean_runtime_ms / batch_size]
            df.loc[len(df)] = new_row

    df = df.sort_values(by=["method", "number of solutions"])

    print(df)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for method_name in method_names:
        df_method = df[df["method"] == method_name]
        n_solutions = df_method["number of solutions"]
        total_runtime_ms = df_method["total runtime (ms)"]
        std = df_method["runtime std"]
        ax.plot(n_solutions, total_runtime_ms, label=method_name)
        ax.fill_between(n_solutions, total_runtime_ms - std, total_runtime_ms + std, alpha=0.2)

    ax.set_xlabel("Number of solutions")
    ax.set_ylabel("Total runtime (ms)")
    ax.legend()
    fig.savefig("scripts/batch_fk_runtime.pdf", bbox_inches="tight")
    plt.show()
