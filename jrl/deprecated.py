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