from time import time

from jrl.config import DEVICE, DEFAULT_TORCH_DTYPE
from jrl.robots import FetchArm, Panda

import torch

torch.set_default_dtype(DEFAULT_TORCH_DTYPE)
torch.set_default_device(DEVICE)

"""python scripts/capsule_collision_stats.py
"""


if __name__ == "__main__":
    always_colliding_pct = 0.95  # value used in moveit
    never_colliding_pct = 0.001

    # Note: you need to manually comment out the collision pairs in 'ignored_collision_pairs' in __init__()
    # robot = FetchArm()
    robot = Panda()

    link_names = list(robot._collision_capsules_by_link.keys())
    n_pairs = robot._collision_idx0.numel()

    print()
    for idx in robot._capsule_idx_to_link_idx:
        print(f"  idx.item() {idx}:", link_names[robot._capsule_idx_to_link_idx[idx].item()])
    print("\nrobot._collision_idx0:", robot._collision_idx0, robot._collision_idx0.numel())
    print("robot._collision_idx1:", robot._collision_idx1, robot._collision_idx1.numel())

    n = 100000
    configs_random = torch.tensor(robot.sample_joint_angles(n, joint_limit_eps=0.0), dtype=DEFAULT_TORCH_DTYPE)
    collisions = robot.self_collision_distances_batch(configs_random)
    colliding = collisions < 0

    collision_counter = {
        (link_names[robot._collision_idx0[j]], link_names[robot._collision_idx1[j]]): 0 for j in range(n_pairs)
    }
    print("collisions:")
    print(collisions.shape)

    def print_collision_counter(ii):
        print(f"\ni: {ii}")
        for k, v in collision_counter.items():
            print(f"  {k}: {v}")

    t0 = time()
    for i in range(n):
        for j in range(n_pairs):
            link_idx0 = robot._collision_idx0[j]
            link_idx1 = robot._collision_idx1[j]

            if colliding[i, j]:
                collision_counter[(link_names[robot._collision_idx0[j]], link_names[robot._collision_idx1[j]])] += 1

        if i % 1000 == 0:
            print_collision_counter(i)
    print_collision_counter(i)

    never_colliding = [pair for pair in collision_counter if collision_counter[pair] / n < never_colliding_pct]
    always_colliding = [pair for pair in collision_counter if collision_counter[pair] / n > always_colliding_pct]
    sometimes_colliding = [
        pair for pair in collision_counter if not (pair in never_colliding or pair in always_colliding)
    ]
    for p in never_colliding:
        assert p not in always_colliding, f"error, {p} is in never_colliding and always_colliding"
        assert p not in sometimes_colliding, f"error, {p} is in never_colliding and sometimes_colliding"
    for p in always_colliding:
        assert p not in sometimes_colliding, f"error, {p} is in always_colliding and sometimes_colliding"

    print("\n=================================")
    print("    ==       Results       ==  ")

    print("\nNever colliding:")
    for p in never_colliding:
        print(f"  {p}:\t{collision_counter[p]}/{n}")

    print("\nAlways colliding:")
    for p in always_colliding:
        print(f"  {p}:\t{collision_counter[p]}/{n}")

    print("\nSometimes colliding:")
    for p in sometimes_colliding:
        print(f"  {p}:\t{collision_counter[p]}/{n}")

    assert len(never_colliding) + len(always_colliding) + len(sometimes_colliding) == n_pairs, (
        "Error - len(never_colliding) + len(always_colliding) + len(sometimes_colliding) != n_pairs"
        f" ({len(never_colliding)} + {len(always_colliding)} + {len(sometimes_colliding)} != {n_pairs})"
    )

    print(f"\nevaluated {n} configs in {time() - t0} seconds")
