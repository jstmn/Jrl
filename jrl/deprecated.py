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











########################################################################################################################
########################################################################################################################
#                                          QP Collision Checking

class Robot:
    def __init__(self):
        # ...
        self._collision_capsules = None
        self._capsule_idx_to_link_idx = None
        self._collision_idx0 = None
        self._collision_idx1 = None
        if self._collision_capsules_by_link is not None:
            (
                self._collision_capsules,
                self._capsule_idx_to_link_idx,
                self._collision_idx0,
                self._collision_idx1,
            ) = _generate_self_collision_pairs(
                self._collision_capsules_by_link,
                self._end_effector_kinematic_chain,
                ignored_collision_pairs,
                additional_link=self._additional_link_name,
                additional_link_lca_joint=(
                    self._additional_link_lca_joint if self._additional_link_name is not None else None
                ),
            )

        self._ignored_collision_pairs = ignored_collision_pairs + [(l2, l1) for l1, l2 in ignored_collision_pairs]


    def self_collision_distances(self, x: torch.Tensor, use_qpth: bool = False) -> torch.Tensor:
        """Returns the distance between all valid collision pairs of the robot
        for each joint angle vector in x

        Args:
            x (torch.Tensor): [n x ndofs] joint angle vectors

        Returns:
            torch.Tensor: [n x n_pairs] distances
        """
        # Capsule and joint indices are offset by 1 to make room for the base
        # link.
        batch_size = x.shape[0]
        base_T_links = self.forward_kinematics(x, return_full_link_fk=True, out_device=x.device, dtype=x.dtype)
        T1s = base_T_links[:, self._collision_idx0, :, :].reshape(-1, 4, 4)
        T2s = base_T_links[:, self._collision_idx1, :, :].reshape(-1, 4, 4)
        c1s = self._collision_capsules[self._collision_idx0, :].expand(batch_size, -1, -1).reshape(-1, 7)
        c2s = self._collision_capsules[self._collision_idx1, :].expand(batch_size, -1, -1).reshape(-1, 7)

        dists = capsule_capsule_distance_batch(c1s, T1s, c2s, T2s, use_qpth).reshape(batch_size, -1)

        return dists

    def self_collision_distances_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the jacobian of the self collision distance function with respect to the joint angles.

        Args:
            x (torch.Tensor): [n x ndofs] joint angle vectors

        Returns:
            torch.Tensor: [n x n_pairs x ndofs] jacobian
        """
        nbatch = x.shape[0]
        ndofs = x.shape[1]

        with torch.autograd.forward_ad.dual_level():
            dual_input = torch.autograd.forward_ad.make_dual(
                x.unsqueeze(1).expand(nbatch, ndofs, ndofs).reshape(nbatch * ndofs, ndofs).clone(),
                torch.eye(ndofs, device=x.device).expand(nbatch, ndofs, ndofs).reshape(-1, ndofs).clone(),
            )
            dual_output = self.self_collision_distances(dual_input)
            J = torch.autograd.forward_ad.unpack_dual(dual_output).tangent
            ndists = J.shape[1]
            J = J.reshape(nbatch, ndofs, ndists).permute(0, 2, 1)

            return J

    def env_collision_distances(self, x: torch.Tensor, cuboid: torch.Tensor, Tcuboid: torch.Tensor) -> torch.Tensor:
        """Returns the distance between the robot collision capsules and the environment cuboid obstacle for each joint 
        angle vector in x.

        Args:
            x (torch.Tensor): [n x ndofs] joint angle vectors
            cuboid (torch.Tensor): [6] cuboid xyz min and xyz max
            Tcuboid (torch.Tensor): [4 x 4] cuboid poses

        Returns:
            torch.Tensor: [n x n_capsules] distances
        """

        batch_size = x.shape[0]
        base_T_links = self.forward_kinematics(x, return_full_link_fk=True, out_device=x.device, dtype=x.dtype)
        Tcapsules = base_T_links.reshape(-1, 4, 4)
        big_batch_size = Tcapsules.shape[0]
        capsules = self._collision_capsules.expand(batch_size, -1, -1).reshape(-1, 7)
        Tcuboid = Tcuboid.expand(big_batch_size, 4, 4)
        cuboid = cuboid.expand(big_batch_size, 6)

        dists = capsule_cuboid_distance_batch(capsules, Tcapsules, cuboid, Tcuboid).reshape(batch_size, -1)

        return dists

    def env_collision_distances_jacobian(
        self, x: torch.Tensor, cuboid: torch.Tensor, Tcuboid: torch.Tensor
    ) -> torch.Tensor:
        nbatch = x.shape[0]
        ndofs = x.shape[1]

        with torch.autograd.forward_ad.dual_level():
            dual_input = torch.autograd.forward_ad.make_dual(
                x.unsqueeze(1).expand(nbatch, ndofs, ndofs).reshape(nbatch * ndofs, ndofs).clone(),
                torch.eye(ndofs, device=x.device).expand(nbatch, ndofs, ndofs).reshape(-1, ndofs).clone(),
            )
            dual_output = self.env_collision_distances(dual_input, cuboid, Tcuboid)
            J = torch.autograd.forward_ad.unpack_dual(dual_output).tangent
            ndists = J.shape[1]
            J = J.reshape(nbatch, ndofs, ndists).permute(0, 2, 1)

            return J

def _generate_self_collision_pairs(
    collision_capsules_by_link: Dict[str, torch.Tensor],
    joint_chain: List[Joint],
    ignored_collision_pairs: List[Tuple[str, str]],
    additional_link: Optional[str] = None,
    additional_link_lca_joint: Optional[Joint] = None,
):
    """
    Generate collision pairs from collision capsules and joint chain. Adjacent links in the joint chain are allowed to
    collide.

    Returns capsules and idx0, idx1, such that capsules[idx0[i]] and capsules[idx1[i]] must be checked for collision.

    Args:
        joint_chain (List[Joint]): Joint chain of the robot.
        ignored_collision_pairs (List[Tuple[str, str]], optional): List of collision pairs to ignore. Defaults to [].
        additional_link (Optional[str]): An optional additional link to add to the collision pairs. Defaults to None.
        additional_link_lca_joint (Optional[Joint]): The artificial joint that connects the additional link to the
                                                        joint_chain

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: capsules, idx0, idx1
    """
    ignored_collision_set = set(tuple(sorted(pair)) for pair in ignored_collision_pairs)
    for link_name, capsule in collision_capsules_by_link.items():
        if capsule is None:
            for other_link_name in collision_capsules_by_link.keys():
                ignored_collision_set.add(tuple(sorted((link_name, other_link_name))))
            collision_capsules_by_link[link_name] = torch.tensor(
                [0, 0, 0, 0, 0, 0.01, 0.001], device=DEVICE, dtype=DEFAULT_TORCH_DTYPE
            )

    link_name_to_idx = {}
    capsule_idx_to_joint_idx = []
    capsules = []
    for i, joint in enumerate(joint_chain):
        if i == 0 and joint.parent in collision_capsules_by_link:
            capsules.append(collision_capsules_by_link[joint.parent])
            link_name_to_idx[joint.parent] = 0
            capsule_idx_to_joint_idx.append(i)

        if joint.child in collision_capsules_by_link:
            capsules.append(collision_capsules_by_link[joint.child])
            link_name_to_idx[joint.child] = i + 1
            capsule_idx_to_joint_idx.append(i + 1)

        ignored_collision_set.add(tuple(sorted((joint.parent, joint.child))))

    # Add in the additional link
    if additional_link is not None:
        capsules.append(collision_capsules_by_link[additional_link])
        idx = capsule_idx_to_joint_idx[-1] + 1
        link_name_to_idx[additional_link] = idx
        capsule_idx_to_joint_idx.append(idx)
        ignored_collision_set.add(tuple(sorted((additional_link_lca_joint.parent, additional_link_lca_joint.child))))

    idx0, idx1 = [], []
    link_names = list(collision_capsules_by_link.keys())
    for i in range(len(link_names)):
        for j in range(i + 1, len(link_names)):
            if (link_names[i], link_names[j]) in ignored_collision_set:
                continue
            if (link_names[j], link_names[i]) in ignored_collision_set:
                continue
            idx0.append(link_name_to_idx[link_names[i]])
            idx1.append(link_name_to_idx[link_names[j]])

    return (
        torch.stack(capsules, dim=0),
        torch.tensor(capsule_idx_to_joint_idx, dtype=torch.long, device=DEVICE),
        torch.tensor(idx0, dtype=torch.long, device=DEVICE),
        torch.tensor(idx1, dtype=torch.long, device=DEVICE),
    )


class QP:
    def __init__(self, Q, p, G, h, A=None, b=None):
        """Solve a quadratic program of the form
        min 1/2 x^T Q x + p^T x
        s.t. Gx <= h, Ax = b

        Args:
            Q (torch.Tensor): [nbatch x dim x dim] positive semidefinite matrix
            p (torch.Tensor): [nbatch x dim] vector
            G (torch.Tensor): [nbatch x nc x dim] matrix
            h (torch.Tensor): [nbatch x nc] vector
            A (torch.Tensor, optional): [nbatch x neq x dim] matrix. Defaults to
            None.
            b (torch.Tensor, optional): [nbatch x neq] vector. Defaults to None.

        Raises:
            NotImplementedError: Equality constraints not implemented

        Returns:
            torch.Tensor: [nbatch x dim] solution
        """
        self.nbatch = Q.shape[0]
        self.dim = Q.shape[1]
        self.nc = G.shape[1]
        self.Q = Q
        assert len(p.shape) == 2, f"p.shape: {p.shape}"
        self.p = p.unsqueeze(2)
        self.G = G
        assert len(h.shape) == 2, f"h.shape: {h.shape}"
        self.h = h.unsqueeze(2)

        assert (
            self.nbatch == self.p.shape[0] == self.G.shape[0] == self.h.shape[0] == self.Q.shape[0]
        ), f"{self.nbatch}, {self.p.shape[0]}, {self.G.shape[0]}, {self.h.shape[0]}, {self.Q.shape[0]}"
        assert (
            self.dim == self.p.shape[1] == self.G.shape[2] == self.Q.shape[1] == self.Q.shape[2]
        ), f"{self.dim}, {self.p.shape[1]}, {self.G.shape[2]}, {self.Q.shape[1]}, {self.Q.shape[2]}"
        assert self.nc == self.G.shape[1] == self.h.shape[1]

        if A is not None or b is not None:
            raise NotImplementedError("Equality constraints not implemented")
        self.A = A
        self.b = b

    def solve(self, trace=False, iterlimit=None):
        x = torch.zeros((self.nbatch, self.dim, 1), dtype=torch.float32, device=self.Q.device)
        if trace:
            trace = [x]
        working_set = torch.zeros((self.nbatch, self.nc, 1), dtype=torch.bool, device=x.device)
        converged = torch.zeros((self.nbatch, 1, 1), dtype=torch.bool, device=x.device)

        if iterlimit is None:
            iterlimit = 10 * self.nc

        iterations = 0
        while not torch.all(converged):
            A = self.G * working_set
            b = self.h * working_set

            g = self.Q.bmm(x) + self.p
            h = A.bmm(x) - b

            AT = A.transpose(1, 2)
            AQinvAT = A.bmm(torch.linalg.solve(self.Q, AT))
            rhs = h - A.bmm(torch.linalg.solve(self.Q, g))
            diag = torch.diag_embed(~working_set.squeeze(dim=2))
            lmbd = torch.linalg.solve(AQinvAT + diag, rhs)
            p = torch.linalg.solve(self.Q, -AT.bmm(lmbd) - g)

            at_ws_sol = torch.linalg.norm(p, dim=1, keepdim=True) < 1e-3

            lmbdmin, lmbdmin_idx = torch.min(
                torch.where(
                    at_ws_sol & working_set,
                    lmbd,
                    torch.inf,
                ),
                dim=1,
                keepdim=True,
            )

            converged = converged | (at_ws_sol & (lmbdmin >= 0))

            # remove inverted constraints from working set
            working_set = working_set.scatter(
                1,
                lmbdmin_idx,
                (~at_ws_sol | (lmbdmin >= 0)) & working_set.gather(1, lmbdmin_idx),
            )

            # check constraint violations
            mask = ~at_ws_sol & ~working_set & (self.G.bmm(p) > 0)
            alpha, alpha_idx = torch.min(
                torch.where(
                    mask,
                    (self.h - self.G.bmm(x)) / (self.G.bmm(p)),
                    torch.inf,
                ),
                dim=1,
                keepdim=True,
            )

            # add violated constraints to working set
            working_set = working_set.scatter(
                1,
                alpha_idx,
                working_set.gather(1, alpha_idx) | (~at_ws_sol & (alpha < 1)),
            )
            alpha = torch.clamp(alpha, max=1)

            # update solution
            x = x + alpha * p * (~at_ws_sol & ~converged)
            if trace:
                trace.append(x.squeeze(2))

            iterations += 1
            if iterations > iterlimit:
                # Qs = self.Q[~converged.flatten()]
                # ps = self.p[~converged.flatten()]
                # Gs = self.G[~converged.flatten()]
                # hs = self.h[~converged.flatten()]
                # xs = x[~converged.flatten()]
                # for i in range(torch.sum(~converged)):
                #     print(f"Qs[{i}]:\n{Qs[i]}")
                #     print(f"ps[{i}]:\n{ps[i]}")
                #     print(f"Gs[{i}]:\n{Gs[i]}")
                #     print(f"hs[{i}]:\n{hs[i]}")
                #     print(f"xs[{i}]:\n{xs[i]}")
                raise RuntimeError(
                    f"Failed to converge in {iterlimit} iterations\n\n{torch.sum(~converged).item()} out of"
                    f" {self.nbatch} not converged"
                )

        if trace:
            return x.squeeze(2), trace

        return x.squeeze(2)

