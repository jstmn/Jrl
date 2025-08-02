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


    def inverse_kinematics_step_levenburg_marquardt_cholesky(
        self,
        target_poses: torch.Tensor,
        xs_current: torch.Tensor,
        lambd: float = 0.0001,
        alpha: float = 1.0,
        alphas: Optional[torch.Tensor] = None,
        clamp_to_joint_limits: bool = True,
    ) -> torch.Tensor:
        """Perform a levenburg-marquardt optimization step."""
        n = xs_current.shape[0]
        eye = torch.eye(self.ndof, device=xs_current.device)[None, :, :].repeat(n, 1, 1)

        # Get current error
        current_poses = self.forward_kinematics(xs_current, out_device=xs_current.device, dtype=xs_current.dtype)
        # TODO: Use cat instead of creating a new tensor
        pose_errors = torch.zeros((n, 6, 1), device=xs_current.device, dtype=xs_current.dtype)  # [n 6 1]
        for i in range(3):
            pose_errors[:, i + 3, 0] = target_poses[:, i] - current_poses[:, i]

        # TODO: implement, test, compare runtime for quaternion_difference_to_rpy()
        # rotation_error_rpy = quaternion_difference_to_rpy(target_poses[:, 3:], current_poses[:, 3:])
        current_pose_quat_inv = quaternion_inverse(current_poses[:, 3:7])
        rotation_error_quat = quaternion_product(target_poses[:, 3:], current_pose_quat_inv)
        rotation_error_rpy = quaternion_to_rpy(rotation_error_quat)
        pose_errors[:, 0:3, 0] = rotation_error_rpy  #

        J_batch = torch.tensor(
            self.jacobian_batch_np(np.array(xs_current.detach().cpu())),
            device=xs_current.device,
            dtype=xs_current.dtype,
        )  # [n 6 ndof]
        assert J_batch.shape == (n, 6, self.ndof)
        J_batch_T = torch.transpose(J_batch, 1, 2)  # [n ndof 6]
        assert J_batch_T.shape == (
            n,
            self.ndof,
            6,
        ), f"error, J_batch_T: {J_batch_T.shape}, should be {(n, self.ndof, 6)}"

        # Solve (J_batch_T^T*J_batch_T + lambd*I)*delta_X = J_batch_T*pose_errors
        # From wikipedia (https://en.wikipedia.org/wiki/Cholesky_decomposition)
        #  Problem: solve Ax=b
        #  Solution:
        #    1. find L s.t. A = L*L^T
        #    2. solve L*y = b for y by forward substitution
        #    3. solve L^T*x = y for y by backward substitution
        # eye = torch.eye(n * ndof, dtype=opt_state.x.dtype, device=opt_state.x.device)
        # eye = torch.eye(n * ndof)
        # J_T = torch.transpose(J, 0, 1)
        # A = torch.matmul(J_T, J) + lambd * eye  # [n*ndof x n*ndof]
        # b = torch.matmul(J_T, r)
        # L = torch.linalg.cholesky(A, upper=False)
        # y = torch.linalg.solve_triangular(L, b, upper=False)
        # delta_x = torch.linalg.solve_triangular(L.T, y, upper=True).reshape((n, ndof))

        eye = torch.eye(self.ndof, device=xs_current.device, dtype=xs_current.dtype)[None, :, :].repeat(n, 1, 1)
        assert eye.shape == (n, self.ndof, self.ndof)

        A = torch.bmm(J_batch_T, J_batch) + lambd * eye  # [n ndof ndof]
        assert A.shape == (n, self.ndof, self.ndof)

        b = torch.bmm(J_batch_T, pose_errors)  # [n x ndof x 6] * [n x 6 x 1] = [n x ndof x 1]
        assert b.shape == (n, self.ndof, 1)

        L = torch.linalg.cholesky(A, upper=False)  # [n ndof ndof]
        assert L.shape == (n, self.ndof, self.ndof)

        y = torch.linalg.solve_triangular(L, b, upper=False)  # [n ndof 1]
        assert y.shape == (n, self.ndof, 1)

        L_T = L.transpose(-2, -1)  # Explicitly transpose to ensure correct shape
        assert L_T.shape == (n, self.ndof, self.ndof), f"L_T.shape: {L_T.shape}, should be {(n, self.ndof, self.ndof)}"
        delta_x = torch.linalg.solve_triangular(L_T, y, upper=True)  # [n ndof 1]

        # lfs_A = torch.bmm(J_batch_T, J_batch) + lambd * eye  # [n ndof ndof]
        # rhs_B = torch.bmm(J_batch_T, pose_errors)  # [n ndof 1]
        # delta_x = torch.linalg.solve(lfs_A, rhs_B)  # [n ndof 1]

        if alphas is not None:
            assert alphas.shape == (n, 1)
            xs_updated = xs_current + alphas * torch.squeeze(delta_x)
        else:
            xs_updated = xs_current + alpha * torch.squeeze(delta_x)

        if clamp_to_joint_limits:
            return self.clamp_to_joint_limits(xs_updated)
        return xs_updated

    # TODO: Enforce joint limits
    def inverse_kinematics_step_jacobian_pinv(
        self,
        target_poses: torch.Tensor,
        xs_current: torch.Tensor,
        alpha: float = 0.25,
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
        assert xs_current.device == target_poses.device, (
            f"xs_current and target_poses must be on the same device (got {xs_current.device} and {target_poses.device})"
        )
        n = target_poses.shape[0]

        # Get the jacobian of the end effector with respect to the current joint angles
        J = torch.tensor(
            self.jacobian_batch_np(xs_current.detach().cpu().numpy()),
            device="cpu",
            dtype=dtype,
        )
        J_pinv = torch.linalg.pinv(J)  # Jacobian pseudo-inverse
        J_pinv = J_pinv.to(xs_current.device)

        # Run the xs_current through FK to get their realized poses
        current_poses = self.forward_kinematics(xs_current, out_device=xs_current.device, dtype=dtype)
        assert current_poses.shape == target_poses.shape, (
            f"current_poses.shape != target_poses.shape ({current_poses.shape} != {target_poses.shape})"
        )

        # Fill out `pose_errors` - the matrix of positional and rotational for each row (rotational error is in rpy)
        pose_errors = torch.zeros((n, 6, 1), device=xs_current.device, dtype=dtype)
        for i in range(3):
            pose_errors[:, i + 3, 0] = target_poses[:, i] - current_poses[:, i]

        current_pose_quat_inv = quaternion_inverse(current_poses[:, 3:7])
        rotation_error_quat = quaternion_product(target_poses[:, 3:], current_pose_quat_inv)
        rotation_error_rpy = quaternion_to_rpy(rotation_error_quat)
        pose_errors[:, 0:3, 0] = rotation_error_rpy  #

        if torch.isnan(pose_errors).sum() > 0:
            for row_i in range(pose_errors.shape[0]):
                if torch.isnan(pose_errors[row_i]).sum() > 0:
                    print(f"\npose_errors[{row_i}] contains NaNs")
                    print(f"target_pose:  {target_poses[row_i].data}")
                    print(f"current_pose: {current_poses[row_i].data}")
                    print(f"pose_error:   {pose_errors[row_i, :, 0].data}")
        assert torch.isnan(pose_errors).sum() == 0, (
            f"pose_errors contains NaNs ({torch.isnan(pose_errors).sum()} of {pose_errors.numel()})"
        )

        # tensor dimensions: [batch x ndofs x 6] * [batch x 6 x 1] = [batch x ndofs x 1]
        delta_x = J_pinv @ pose_errors
        xs_updated = xs_current + alpha * delta_x[:, :, 0]

        assert torch.isnan(xs_updated).sum() == 0, "xs_updated contains NaNs"
        assert xs_current.device == xs_updated.device
        xs_updated = self.clamp_to_joint_limits(xs_updated)
        return xs_updated

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
        assert xs_current.device == target_poses.device, (
            f"xs_current and target_poses must be on the same device (got {xs_current.device} and {target_poses.device})"
        )

        # New graph
        xs_current = xs_current.detach()
        xs_current.requires_grad = True

        # Run the xs_current through FK to get their realized poses
        current_poses = self.forward_kinematics(xs_current, out_device=xs_current.device, dtype=dtype)
        assert current_poses.shape == target_poses.shape, (
            f"current_poses.shape != target_poses.shape ({current_poses.shape} != {target_poses.shape})"
        )

        t_err = target_poses[:, 0:3] - current_poses[:, 0:3]
        R_err = geodesic_distance_between_quaternions(target_poses[:, 3:7], current_poses[:, 3:7])
        loss = torch.sum(t_err**2) + torch.sum(R_err**2)
        loss.backward()

        xs_updated = xs_current - alpha * xs_current.grad

        assert torch.isnan(xs_updated).sum() == 0, "xs_updated contains NaNs"
        assert xs_current.device == xs_updated.device
        xs_updated = self.clamp_to_joint_limits(xs_updated)
        return xs_updated.detach()