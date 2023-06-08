import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import jkinpylib
import time
import tqdm


def main():
    robot = jkinpylib.robots.Fetch()
    timing_df = pd.DataFrame(columns=["nbatch", "ntrials", "time"])
    batch_sizes = [1, 10, 100, 200, 300, 500]
    for nbatch in tqdm.tqdm(batch_sizes):
        ntrials = 10
        for i in range(ntrials):
            x = torch.tensor(robot.sample_joint_angles(nbatch), dtype=torch.float32)
            t0 = time.time()
            dists = robot.self_collision_distances_jacobian_batch(x)
            t1 = time.time()
            timing_df.loc[len(timing_df)] = [nbatch, ntrials, t1 - t0]

    fig, ax = plt.subplots()
    # plot mean and filled std of timing vs batch size
    ax.plot(timing_df.groupby("nbatch").mean()["time"], label="mean")
    ax.fill_between(
        timing_df.groupby("nbatch").mean()["time"].index,
        timing_df.groupby("nbatch").mean()["time"] - timing_df.groupby("nbatch").std()["time"],
        timing_df.groupby("nbatch").mean()["time"] + timing_df.groupby("nbatch").std()["time"],
        alpha=0.3,
        label="std",
    )

    ax.set_title(f"self collision distance jacobian batch (Fetch, {dists.device})")
    ax.set_xlabel("batch size")
    ax.set_ylabel("time (s)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("benchmark_collision.png")
    plt.savefig("benchmark_collision.pdf")
    plt.show()

    print("device:", dists.device)


if __name__ == "__main__":
    main()
