import torch
import numpy as np

from jrl.robots import Panda
from jrl.utils import evenly_spaced_colors
from jrl.math_utils import geodesic_distance_between_quaternions

import matplotlib.pyplot as plt


def main(robot):
    noise_scales = [0.1, 0.5, 1.0, 3.1415]
    fig, axs = plt.subplots(len(noise_scales), 4, figsize=(20, 20))
    fig.suptitle(
        "Step-size vs. end effector pose error convergence. Step direction calculated with Levenberg Marquardt"
    )
    fig.tight_layout(pad=3)
    # n = 500
    n = 200

    # alphas = [0.0, 0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]
    alphas = np.linspace(0.0, 1.5, 20)
    colors = evenly_spaced_colors(len(alphas))

    for row_idx, noise_scale in enumerate(noise_scales):

        rand_scale = noise_scale
        qs_0, poses_0 = robot.sample_joint_angles_and_poses(n, return_torch=True)
        qs_pert = robot.clamp_to_joint_limits(qs_0 + rand_scale * (torch.rand_like(qs_0) - 0.5))

        # get summary stats
        ee = robot.forward_kinematics(qs_pert)
        mean_pos_error = (torch.norm(ee[:, 0:3] - poses_0[:, 0:3], dim=1).mean() * 100).item()
        mean_rot_error = (
            torch.rad2deg(geodesic_distance_between_quaternions(ee[:, 3 : 3 + 4], poses_0[:, 3 : 3 + 4])).mean().item()
        )
        # print("q difference:", torch.rad2deg(qs_pert - qs_0))

        axl = axs[row_idx, 0]
        axr = axs[row_idx, 1]
        axrr = axs[row_idx, 2]
        axrrr = axs[row_idx, 3]
        axl.set_title(
            f"noise_scale: {noise_scale}, mean_pos_error: {mean_pos_error:.3f} [cm], mean_rot_error:"
            f" {mean_rot_error:.3f} [deg]"
        )
        axl.set_xlabel("")
        axl.set_xlabel("")
        axr.set_xlabel("")
        axrr.set_xlabel("Alpha")
        axrrr.set_xlabel("Alpha")
        axl.set_ylabel("Error [cm]")
        axr.set_ylabel("Error [deg]")
        axrr.set_ylabel("Error [cm]")
        axrrr.set_ylabel("Count, best alpha")
        axl.grid("both", alpha=0.2)
        axr.grid("both", alpha=0.2)
        axrr.grid("both", alpha=0.2)
        axrrr.grid("both", alpha=0.2)

        max_n_Qs = 10

        # colors = matplotlib.colors.TABLEAU_COLORS
        def plot_qs(qs_, label, i, large_dot=False, color=None):
            _n = min(n, max_n_Qs)
            color = colors[i] if color is None else color
            ee = robot.forward_kinematics(qs_)
            pos_errors = torch.norm(ee[:, 0:3] - poses_0[:, 0:3], dim=1) * 100
            rot_errors = torch.rad2deg(geodesic_distance_between_quaternions(ee[:, 3 : 3 + 4], poses_0[:, 3 : 3 + 4]))
            x_offset = i / 30.0
            axl.scatter(
                np.arange(0, _n) + x_offset,
                pos_errors.cpu().numpy()[:_n],
                label=label,
                s=75.0 if large_dot else None,
                color=color,
            )
            axr.scatter(
                np.arange(0, _n) + x_offset,
                rot_errors.cpu().numpy()[:_n],
                label=label,
                s=75.0 if large_dot else None,
                color=color,
            )
            return (
                pos_errors.mean().item(),
                pos_errors.std().item(),
                rot_errors.mean().item(),
                rot_errors.std().item(),
                pos_errors,
            )

        plot_qs(qs_pert, "qs original", 0, large_dot=True, color="black")

        mean_pos_errors = []
        std_pos_errors = []
        all_pos_errors = torch.zeros((len(alphas), n))
        for i, alpha in enumerate(alphas):
            mean, std, _, _, pos_errors = plot_qs(
                robot.inverse_kinematics_step_levenburg_marquardt(poses_0, qs_pert, alpha=alpha),
                f"LM: {float(alpha):.2f}",
                i,
            )
            mean_pos_errors.append(mean)
            std_pos_errors.append(std)
            all_pos_errors[i] = pos_errors
        assert abs(mean_pos_error - mean_pos_errors[0]) < 1e-5

        # calculate best alphas for each config
        min_values, min_indices = torch.min(all_pos_errors, dim=0)
        assert min_values.numel() == n
        best_alphas = [alphas[j] for j in min_indices]
        axrr.fill_between(
            alphas,
            np.array(mean_pos_errors) - np.array(std_pos_errors),
            np.array(mean_pos_errors) + np.array(std_pos_errors),
            alpha=0.1,
            label="std",
            color="b",
        )
        axrr.plot(alphas, mean_pos_errors, color="b")
        axrr.scatter(alphas, mean_pos_errors, color="b")
        axrr.set_ylim(0, max(mean_pos_errors) * 1.5)
        axrr.plot([0, max(alphas)], [mean_pos_error, mean_pos_error], color="k", linestyle="dashed", alpha=0.5)

        axrrr.hist(best_alphas, color="black", alpha=0.9, bins=int(n / 10))

    # axl.legend()
    axs[0, 0].legend(prop={"size": 6})
    plt.show()


# QUESTION: with an efficient line search method, is the better step size worth the extra FK time?


""" python scripts/step_size_line_search_experiment.py
"""

if __name__ == "__main__":
    robot = Panda()
    main(robot)
