import warp as wp

wp.init()


@wp.kernel
def _geodesic_distance_quaternions(
    q1: wp.array(dtype=wp.quatf), q2: wp.array(dtype=wp.quatf), dist: wp.array(dtype=float)
):
    tid = wp.tid()
    quat1 = q1[tid]
    quat2 = q2[tid]
    # dist = 2 * arccos( 2*<q1, q2> - 1 )
    dot = wp.dot(quat1, quat2)
    # Note: for wp.acos(...) "Inputs are automatically clamped to [-1.0, 1.0]. See https://nvidia.github.io/warp/_build/html/modules/functions.html#warp.acos
    dist[tid] = 2.0 * wp.acos(dot)


def geodesic_distance_between_quaternions_warp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Given rows of quaternions q1 and q2, compute the geodesic distance between each
    """
    # Note: Decreasing this value to 1e-8 greates NaN gradients for nearby quaternions.
    assert not q1.requires_grad and not q2.requires_grad
    q1_wp = wp.from_torch(q1, dtype=wp.quatf)
    q2_wp = wp.from_torch(q2, dtype=wp.quatf)
    dist_wp = wp.zeros(q1.shape[0], dtype=float, device=str(q1.device))
    wp.launch(kernel=_geodesic_distance_quaternions, dim=len(q1), inputs=[q1_wp, q2_wp, dist_wp], device=str(q1.device))
    return wp.to_torch(dist_wp).to(q1.device)

    def inverse_kinematics_autodiff_single_step_batch_pt(
        self,
        target_poses: torch.Tensor,
        xs_current: torch.Tensor,
        alpha: float = 0.10,
        dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
    ) -> torch.Tensor:
        """Perform a single inverse kinematics step on a batch of joint angle vectors using pytorch.

        Notes:
            1. `target_poses` and `xs_current` need to be on the same device.
            2. the returned tensor will be on the same device as `target_poses` and `xs_current`

        Args:
            target_poses (torch.Tensor): [batch x 7] poses to optimize the joint angles towards.
            xs_current (torch.Tensor): [batch x ndofs] joint angles to start the optimization from.
            alpha (float, optional): Step size for the optimization step. Defaults to 0.25.

        Returns:
            torch.Tensor: Updated joint angles
        """
        assert self._batch_fk_enabled, "_batch_fk_enabled is required for batch_ik, but is disabled for this robot"
        _assert_is_pose_matrix(target_poses)
        _assert_is_joint_angle_matrix(xs_current, self.ndof)
        assert xs_current.shape[0] == target_poses.shape[0]
        assert (
            xs_current.device == target_poses.device
        ), f"xs_current and target_poses must be on the same device (got {xs_current.device} and {target_poses.device})"

        # New graph
        xs_current = xs_current.detach()
        xs_current.requires_grad = True

        # Run the xs_current through FK to get their realized poses
        current_poses = self.forward_kinematics(xs_current, out_device=xs_current.device, dtype=dtype)
        assert (
            current_poses.shape == target_poses.shape
        ), f"current_poses.shape != target_poses.shape ({current_poses.shape} != {target_poses.shape})"

        t_err = target_poses[:, 0:3] - current_poses[:, 0:3]
        R_err = geodesic_distance_between_quaternions(target_poses[:, 3:7], current_poses[:, 3:7])
        loss = torch.sum(t_err**2) + torch.sum(R_err**2)
        loss.backward()

        xs_updated = xs_current - alpha * xs_current.grad

        assert torch.isnan(xs_updated).sum() == 0, "xs_updated contains NaNs"
        assert xs_current.device == xs_updated.device
        xs_updated = self.clamp_to_joint_limits(xs_updated)
        return xs_updated.detach()
