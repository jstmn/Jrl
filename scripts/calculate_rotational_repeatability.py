from math import floor
from jrl.evaluation import pose_errors_cm_deg
from jrl.robots import get_all_robots
from jrl.utils import set_seed
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(suppress=True, linewidth=120)
set_seed()


""" python scripts/calculate_rotational_repeatability.py

This script estimates the rotational repeatability of all robots in the repository. Hardware specifications for robots 
always comes with a _positional_ repeatability value (typically 0.1mm), however the rotational repeatibility is 
typically ommited. This script estimates the rotational repeatibility as a function of the positional repeatibility. The
idea is to find configurations that have positional pose error just within the positional repeatibility of the robot by 
randomly perturbing a configuration. Aggregate statistics of the rotational error of these configs are then calculated 
and reported.
"""

if __name__ == "__main__":
    n_poses = 5000
    n_perturbs = 250
    show_plot = False

    robots = get_all_robots()
    for robot in robots:
        joint_angles, poses = robot.sample_joint_angles_and_poses(n_poses, tqdm_enabled=False)
        max_angular_errors_deg = []

        for q, pose in tqdm(zip(joint_angles, poses), total=n_poses):
            q_last_inbound = []
            for _ in range(n_perturbs):
                pertubation = np.random.random(robot.ndof) - 0.5
                pertubation = 0.00001 * pertubation / np.linalg.norm(pertubation)
                q_perturbed = q.copy()
                while True:
                    q_perturbed_inbound = q_perturbed.copy()
                    q_perturbed += pertubation
                    pos_error_cm, rot_error_deg = pose_errors_cm_deg(
                        robot.forward_kinematics_klampt(q_perturbed[None, :]), pose[None, :], acos_epsilon=1e-30
                    )
                    pos_error_mm = pos_error_cm[0] * 10
                    if pos_error_mm > robot.positional_repeatability_mm:
                        break

                q_last_inbound.append(q_perturbed_inbound)

            q_last_inbound = np.array(q_last_inbound)
            norms = np.linalg.norm(q_last_inbound, axis=1)
            pos_error_cm, rot_error_deg = pose_errors_cm_deg(
                robot.forward_kinematics_klampt(q_last_inbound),
                pose[None, :].repeat(n_perturbs, axis=0),
                acos_epsilon=1e-30,
            )
            pos_error_mm = pos_error_cm * 10

            max_angular_errors_deg.append(np.max(rot_error_deg))

            if show_plot:
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))
                ax: plt.Axes = axs[0]
                ax.scatter(norms, pos_error_mm, label="positional error (cm)")
                ax.set_xlabel("q_perturbed norm")
                ax.set_ylabel("positional error (mm)")
                ax.legend()

                #
                ax: plt.Axes = axs[1]
                ax.scatter(norms, rot_error_deg, label="rotational error (rad)")
                ax.set_xlabel("q_perturbed norm")
                ax.set_ylabel("rotational error (deg)")
                ax.legend()
                plt.show()
                plt.close()

        np.save(
            f"max_angular_errors_deg__{robot.name}__n_poses={n_poses}__n_perturbs={n_perturbs}", max_angular_errors_deg
        )
        print(f"max_angular_errors_deg - {robot.name}:")
        print(" ", np.mean(max_angular_errors_deg))
        print(" ", np.median(max_angular_errors_deg))
        print(" ", np.min(max_angular_errors_deg))
        print(" ", np.max(max_angular_errors_deg))
