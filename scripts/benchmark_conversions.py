from time import time
from typing import Callable
import sys

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from jrl.utils import random_quaternions
from jrl.conversions import geodesic_distance_between_quaternions, geodesic_distance_between_quaternions_old


def fn_mean_std(fn: Callable, k: int):
    runtimes = []
    for _ in range(k):
        t0 = time()
        fn()
        runtimes.append(1000 * (time() - t0))
    return np.mean(runtimes), np.std(runtimes)


""" Example 

python scripts/benchmark_conversions.py


"""


if __name__ == "__main__":
    assert torch.cuda.is_available()

    k = 5
    df = pd.DataFrame(
        columns=["method", "number of solutions", "total runtime (ms)", "runtime std", "runtime per solution (ms)"]
    )
    method_names = [
        "updated - pytorch (cuda)",
        "updated - pytorch (cpu)",
        "updated - numpy",
        "old - pytorch (cuda)",
        "old - pytorch (cpu)",
        "old - numpy",
    ]

    for batch_size in [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
        print(f"Batch size: {batch_size}")

        qs1_cuda = random_quaternions(batch_size, device="cuda")
        qs1_cpu = qs1_cuda.clone().cpu()
        qs1_numpy = qs1_cpu.clone().numpy()
        qs2_cuda = random_quaternions(batch_size, device="cuda")
        qs2_cpu = qs2_cuda.clone().cpu()
        qs2_numpy = qs2_cpu.clone().numpy()

        lambdas = [
            lambda: geodesic_distance_between_quaternions(qs1_cuda, qs2_cuda),
            lambda: geodesic_distance_between_quaternions(qs1_cpu, qs2_cpu),
            lambda: geodesic_distance_between_quaternions(qs1_numpy, qs2_numpy),
            lambda: geodesic_distance_between_quaternions_old(qs1_cuda, qs2_cuda),
            lambda: geodesic_distance_between_quaternions_old(qs1_cpu, qs2_cpu),
            lambda: geodesic_distance_between_quaternions_old(qs1_numpy, qs2_numpy),
        ]
        for lambda_, method_name in zip(lambdas, method_names):
            mean_runtime_ms, std_runtime = fn_mean_std(lambda_, k)
            new_row = [method_name, batch_size, mean_runtime_ms, std_runtime, mean_runtime_ms / batch_size]
            df.loc[len(df)] = new_row

    df = df.sort_values(by=["method", "number of solutions"])

    print(df)

    with open("benchmarking/rotational_distance_runtime_comparison.md", "w") as f:
        f.write("## Rotational distance runtime comparison\n")
        f.write("- updated: `geodesic_distance_between_quaternions()`\n")
        f.write("- old:     `geodesic_distance_between_quaternions_old()`\n")
        cli_input = " ".join(sys.argv)
        f.write(f"\nResults generated with `{cli_input}`\n\n")
        f.write(df.to_markdown())

    # Plot
    max_runtime = -1
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
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
    fig.savefig("benchmarking/rotational_distance_runtime_comparison.pdf", bbox_inches="tight")
    plt.show()
