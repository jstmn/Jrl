from time import time
from typing import Callable
import sys

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from jrl.utils import random_quaternions
from jrl.math_utils import geodesic_distance_between_quaternions


def fn_mean_std(fn: Callable, k: int):
    runtimes = []
    for _ in range(k):
        t0 = time()
        fn()
        runtimes.append(1000 * (time() - t0))
    return np.mean(runtimes), np.std(runtimes)


""" Example 

uv run python scripts/benchmark_math_utils.py


"""


if __name__ == "__main__":
    k = 10
    df = pd.DataFrame(
        columns=["method", "number of solutions", "total runtime (ms)", "runtime std", "runtime per solution (ms)"]
    )
    method_names = [
        "pytorch (cuda)",
        "pytorch (cpu)",
    ]

    for batch_size in [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
        print(f"Batch size: {batch_size}")

        qs1_gpu = random_quaternions(batch_size, device="cuda")
        qs1_cpu = qs1_gpu.clone().cpu()
        qs2_gpu = random_quaternions(batch_size, device="cuda")
        qs2_cpu = qs2_gpu.clone().cpu()

        lambdas = [
            lambda: geodesic_distance_between_quaternions(qs1_gpu, qs2_gpu),
            lambda: geodesic_distance_between_quaternions(qs1_cpu, qs2_cpu),
        ]
        for lambda_, method_name in zip(lambdas, method_names):
            mean_runtime_ms, std_runtime = fn_mean_std(lambda_, k)
            new_row = [method_name, batch_size, mean_runtime_ms, std_runtime, mean_runtime_ms / batch_size]
            df.loc[len(df)] = new_row

    df = df.sort_values(by=["method", "number of solutions"])

    print(df)

    # Plot
    max_runtime = -1
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.grid(alpha=0.2)
    for method_name in method_names:
        df_method = df[df["method"] == method_name]
        n_solutions = df_method["number of solutions"]
        total_runtime_ms = df_method["total runtime (ms)"]
        std = df_method["runtime std"]
        ax.plot(n_solutions, total_runtime_ms, label=method_name)
        ax.fill_between(n_solutions, total_runtime_ms - std, total_runtime_ms + std, alpha=0.2)
        max_runtime = max(max_runtime, total_runtime_ms.to_numpy()[-1])

    ax.set_ylim(-0.1, max_runtime + 0.5)
    ax.set_title("Number of solutions vs runtime for geodesic_distance_between_quaternions()")
    ax.set_xlabel("Number of solutions")
    ax.set_ylabel("Total runtime (ms)")
    ax.legend()
    fig.savefig("rotational_distance_runtime_comparison.pdf", bbox_inches="tight")
    plt.show()
