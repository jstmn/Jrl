import argparse
from time import time
from typing import Callable

import numpy as np
import torch
import pandas as pd

from jkinpylib.utils import to_torch
from jkinpylib.robots import Panda


def fn_mean_std(fn: Callable, k: int):
    runtimes = []
    for _ in range(k):
        t0 = time()
        fn()
        runtimes.append(time() - t0)
    return np.mean(runtimes), np.std(runtimes)


""" Example 

python scripts/benchmark_fk.py


"""


if __name__ == "__main__":
    assert torch.cuda.is_available()

    robot = Panda()
    k = 3
    df = pd.DataFrame(
        columns=["method", "number of solutions", "total runtime (s)", "runtime std", "runtime per solution (ms)"]
    )

    method_names = ["batch_fk - cpu", "batch_fk - cuda", "klampt"]

    for batch_size in [1, 10, 100, 1000, 10000, 100000]:
        print(f"Batch size: {batch_size}")

        x = robot.sample_joint_angles(batch_size)
        x_pt_cpu = to_torch(x.copy()).cpu()
        x_pt_cuda = to_torch(x.copy()).cuda()

        lambdas = [
            lambda: robot.forward_kinematics_batch(x_pt_cpu, out_device="cpu"),
            lambda: robot.forward_kinematics_batch(x_pt_cuda, out_device="cuda"),
            lambda: robot.forward_kinematics_klampt(x),
        ]
        for lambda_, method_name in zip(lambdas, method_names):
            mean_runtime, std_runtime = fn_mean_std(lambda_, k)
            new_row = [method_name, batch_size, mean_runtime, std_runtime, 1000 * (mean_runtime / batch_size)]
            df.loc[len(df)] = new_row

    df = df.sort_values(by=["method", "number of solutions"])

    print(df)
