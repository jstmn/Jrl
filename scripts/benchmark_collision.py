import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import jrl
import time
import tqdm

from jrl.config import DEVICE


def main():
    batch_sizes = [1, 10, 100, 200, 300, 500, 1000, 5000, 10000]
    robot = jrl.robots.Fetch()
    timing_self_df = pd.DataFrame(columns=["nbatch", "ntrials", "time"])
    print("Self collisions")
    for nbatch in tqdm.tqdm(batch_sizes):
        ntrials = 10
        for i in range(ntrials):
            x = torch.tensor(robot.sample_joint_angles(nbatch), dtype=torch.float32, device=DEVICE)
            t0 = time.time()
            dists = robot.self_collision_distances_jacobian(x)
            t1 = time.time()
            timing_self_df.loc[len(timing_self_df)] = [nbatch, ntrials, t1 - t0]

    timing_env_df = pd.DataFrame(columns=["nbatch", "ntrials", "time"])
    print("Environment collisions")
    Tcuboid = torch.eye(4, dtype=torch.float32)
    cuboid = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    for nbatch in tqdm.tqdm(batch_sizes):
        ntrials = 10
        for i in range(ntrials):
            x = torch.tensor(robot.sample_joint_angles(nbatch), dtype=torch.float32, device=DEVICE)
            t0 = time.time()
            dists = robot.env_collision_distances_jacobian(x, cuboid, Tcuboid)
            t1 = time.time()
            timing_env_df.loc[len(timing_env_df)] = [nbatch, ntrials, t1 - t0]

    # plot timing for self and environment next to each other
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title("Self collisions")
    ax[1].set_title("Environment collisions")
    for i, df in enumerate([timing_self_df, timing_env_df]):
        # plot mean and std
        df_mean = df.groupby("nbatch").mean()
        df_std = df.groupby("nbatch").std()
        ax[i].plot(df_mean.index, df_mean["time"], label="mean")
        ax[i].fill_between(
            df_mean.index,
            df_mean["time"] - df_std["time"],
            df_mean["time"] + df_std["time"],
            alpha=0.5,
            label="std",
        )
        ax[i].legend()
        ax[i].set_xlabel("Batch size")
        ax[i].set_ylabel("Time (s)")

    plt.savefig("benchmark_collision_checking.pdf")
    plt.show()

    print("device:", dists.device)


if __name__ == "__main__":
    main()
