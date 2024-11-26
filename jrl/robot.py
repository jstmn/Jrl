from typing import List, Tuple, Optional, Union, Dict
from functools import cached_property

import torch
import numpy as np
import tqdm

from jrl.math_utils import (
    rpy_tuple_to_rotation_matrix,
    single_axis_angle_to_rotation_matrix,
    quaternion_inverse,
    quaternion_product,
    quaternion_to_rpy,
    rotation_matrix_to_quaternion,
    quaternion_norm,
    DEFAULT_TORCH_DTYPE,
)
from jrl.config import DEVICE, PT_NP_TYPE
from jrl.urdf_utils import (
    Joint,
    get_kinematic_chain,
    get_urdf_filepath_w_filenames_updated,
    get_lowest_common_ancestor_link,
    UNHANDLED_JOINT_TYPES,
    merge_fixed_joints_to_one,
)
from jrl.utils import to_torch, _assert_is_2d, _assert_is_pose_matrix, _assert_is_joint_angle_matrix
from jrl.geometry import capsule_capsule_distance_batch, capsule_cuboid_distance_batch



class Robot:
    def __init__(
        self,
        name: str,
        urdf_filepath: str,
        active_joints: List[str],
        base_link: str,
        end_effector_link_name: str,
        ignored_collision_pairs: List[Tuple[str, str]],
        collision_capsules_by_link: Dict[str, torch.Tensor],
        batch_fk_enabled: bool = True,
        verbose: bool = False,
        additional_link_name: Optional[str] = None,
    ):
        """Create a Robot object

        Args:
            name (str): _description_
            urdf_filepath (str): _description_
            active_joints (List[str]): The name of the actuated joints in the kinematic chain that is being represented.
                                        These joints must be along the link chain from the 'base_link' to the
                                        'end_effector_link_name'. Note that all non-fixed joints in this chain that are
                                        not in 'active_joints' will be ignored (by being changed to fixed joints).
            base_link (str): _description_
            end_effector_link_name (str): _description_
            additional_link_name (Optional[str]): Optionally provide the name of an additional link whose pose will be
                                                    returned by forward_kinematics when 'return_full_link_fk' is
                                                    True
            ignored_collision_pairs (List[Tuple[str, str]]): _description_
            collision_capsules_by_link (Optional[Dict[str, torch.Tensor]], optional): _description_. Defaults to None.
            batch_fk_enabled (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to False.
        """
        assert isinstance(name, str)
        assert isinstance(base_link, str)
        assert isinstance(urdf_filepath, str)
        assert isinstance(active_joints, list)
        assert isinstance(end_effector_link_name, str)
        assert isinstance(ignored_collision_pairs, list)
        self._name = name
        self._urdf_filepath = urdf_filepath
        self._base_link = base_link
        self._end_effector_link_name = end_effector_link_name
        self._additional_link_name = additional_link_name
        self._collision_capsules_by_link = collision_capsules_by_link
        self._batch_fk_enabled = batch_fk_enabled
        self._active_joints = active_joints

        # Create the kinematic chain
        self._end_effector_kinematic_chain = get_kinematic_chain(
            self._urdf_filepath,
            self._active_joints,
            self._base_link,
            self._end_effector_link_name,
        )

        # Handle the additional link
        if self._additional_link_name is not None:
            self._additional_link_lca_link = get_lowest_common_ancestor_link(
                self._urdf_filepath, self._end_effector_kinematic_chain, self._active_joints, self._additional_link_name
            )
            self._addl_link_kinematic_chain = get_kinematic_chain(
                self._urdf_filepath, self._active_joints, self._additional_link_lca_link, self._additional_link_name
            )
            self._additional_link_lca_joint = merge_fixed_joints_to_one(self._addl_link_kinematic_chain)

        self._actuated_joint_limits = [
            joint.limits for joint in self._end_effector_kinematic_chain if joint.is_actuated
        ]
        self._actuated_joint_names = [joint.name for joint in self._end_effector_kinematic_chain if joint.is_actuated]
        self._actuated_joint_velocity_limits = [
            joint.velocity_limit for joint in self._end_effector_kinematic_chain if joint.is_actuated
        ]
        assert len(self._active_joints) == self.ndof, (
            f"Error - the number of active joints ({len(self._active_joints)}) does not match the degrees of freedom"
            f" ({self.ndof})."
        )

        # Create and fill cache of fixed rotations between links.
        self._parent_T_joint_cache = {}
        if self._batch_fk_enabled:
            self.forward_kinematics(
                torch.tensor(
                    self.sample_joint_angles(1050),
                    device=DEVICE,
                    dtype=DEFAULT_TORCH_DTYPE,
                )
            )

        self._urdf_filepath_absolute = get_urdf_filepath_w_filenames_updated(self._urdf_filepath)


        if verbose:
            print("\n----------- Robot specs")
            print(f"name: {self.name}")
            print("joints:")
            sjld = 0
            for i, (joint_name, (l, u)) in enumerate(zip(self.actuated_joint_names, self.actuated_joints_limits)):
                print(f"  {i} {joint_name}:\t{round(l, 3)},\t{round(u, 3)}")
                sjld += u - l
            print(f"sum joint range: {round(sjld, 4)} rads")
            print("-----------\n")


    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                                 Properties                                                 ---
    # ---                                                                                                            ---

    @property
    def name(self) -> str:
        """Returns the name of the robot. For example, 'panda'"""
        return self._name

    @property
    def urdf_filepath(self) -> str:
        """Returns the filepath to the urdf file"""
        return self._urdf_filepath

    @property
    def urdf_filepath_absolute(self) -> str:
        """Returns the filepath to the urdf which was rewritten with absolute filepaths to the meshes in urdfs/"""
        return self._urdf_filepath_absolute

    @property
    def end_effector_link_name(self) -> str:
        """Returns the name of the end effector link in the urdf"""
        return self._end_effector_link_name

    @property
    def additional_link(self) -> Optional[str]:
        """Returns the name of the additional link in the urdf"""
        return self._additional_link_name

    @cached_property
    def ndof(self) -> int:
        return sum([1 for joint in self._end_effector_kinematic_chain if joint.is_actuated])

    @property
    def n_collision_pairs(self) -> int:
        """Return the number of capsule-capsule collision pairs of the robot"""
        assert self._collision_capsules is not None, f"collision_capsules not defined for '{self.name}'"
        return self._collision_idx0.numel()

    @property
    def n_joints_in_kinematic_chain(self) -> int:
        """The number of joints in the path from the base_link to the end effector"""
        return len(self._end_effector_kinematic_chain)

    @property
    def actuated_joint_names(self) -> List[str]:
        return self._actuated_joint_names

    @cached_property
    def actuated_joint_types(self) -> List[str]:
        return [joint.joint_type for joint in self._end_effector_kinematic_chain if joint.is_actuated]

    @property
    def actuated_joints_limits(self) -> List[Tuple[float, float]]:
        return self._actuated_joint_limits

    @property
    def actuated_joints_velocity_limits(self) -> List[float]:
        """Measured in rad/s for revolute joints, m/s for prismatic joints"""
        return self._actuated_joint_velocity_limits

    @property
    def actuated_joints_velocity_limits_deg(self) -> List[float]:
        """Measured in deg/s for revolute joints, m/s for prismatic joints"""
        vals = []
        for joint in self._end_effector_kinematic_chain:
            if joint.is_actuated and (joint.joint_type == "revolute" or joint.joint_type == "continuous"):
                vals.append(joint.velocity_limit * 180 / np.pi)
            elif joint.is_actuated and (joint.joint_type == "prismatic"):
                vals.append(joint.velocity_limit)

        assert len(vals) == self.ndof, f"Error, only {len(vals)} in vals, but {self.ndof} degrees of freedom"
        return vals


    @property
    def revolute_joint_idxs(self) -> List[int]:
        return [i for i in range(self.ndof) if self.actuated_joint_types[i] in {"revolute", "continuous"}]

    @property
    def prismatic_joint_idxs(self) -> List[int]:
        return [i for i in range(self.ndof) if self.actuated_joint_types[i] == "prismatic"]

    @property
    def has_prismatic_joints(self) -> bool:
        return len(self.prismatic_joint_idxs) > 0

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             External Functions                                             ---
    # ---                                                                                                            ---

    def split_configs_to_revolute_and_prismatic(self, configs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Returns the values for the values from the revolute, and prismatic joints separately"""
        assert configs.shape[1] == self.ndof
        assert len(self.revolute_joint_idxs) + len(self.prismatic_joint_idxs) == self.ndof
        return configs[:, self.revolute_joint_idxs], configs[:, self.prismatic_joint_idxs]

    def sample_joint_angles(self, n: int, joint_limit_eps: float = 1e-6) -> np.ndarray:
        """Returns a [N x ndof] matrix of randomly drawn joint angle vectors. The joint angles are sampled from the
        range [lower+joint_limit_eps, upper-joint_limit_eps]
        """
        angs = np.random.rand(n, self.ndof)  # between [0, 1)

        # Sample
        for i, (lower, upper) in enumerate(self.actuated_joints_limits):
            # Add eps padding to avoid the joint limits
            upper = upper - joint_limit_eps
            lower = lower + joint_limit_eps
            range_ = upper - lower
            assert range_ > 0
            angs[:, i] *= range_
            angs[:, i] += lower
        return angs

    def sample_joint_angles_and_poses(
        self,
        n: int,
        joint_limit_eps: float = 1e-6,
        only_non_self_colliding: bool = True,
        tqdm_enabled: bool = False,
        return_torch: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a [N x ndof] matrix of randomly drawn joint angle vectors with matching end effector poses."""
        samples = np.zeros((n, self.ndof))
        poses = np.zeros((n, 7))
        internal_batch_size = 5000
        counter = 0

        with tqdm.tqdm(total=n, disable=not tqdm_enabled) as pbar:
            while True:
                samples_i = to_torch(self.sample_joint_angles(internal_batch_size, joint_limit_eps=joint_limit_eps))
                counter0_i = counter

                for i in range(samples_i.shape[0]):
                    sample = samples_i[i]
                    if only_non_self_colliding and self.config_self_collides(sample):
                        continue

                    pose = self.forward_kinematics(sample[None, :])
                    samples[counter] = sample
                    poses[counter] = pose
                    counter += 1
                    pbar.update(1)

                    if counter == n:
                        if return_torch:
                            return to_torch(samples), to_torch(poses)
                        return samples, poses

                if counter0_i == counter:
                    raise RuntimeError(
                        f"Unable to find non self-colliding configs for {self} ({0} /"
                        f" {internal_batch_size} non-self-colliding configs found) - is 'ignored_collision_pairs' set"
                        " correctly for this robot?"
                    )


    def clamp_to_joint_limits(self, x: PT_NP_TYPE):
        """Clamp the given joint angle vectors to the joint limits

        Args:
            x (PT_NP_TYPE): [batch x ndofs] tensor of joint angles
        """
        assert isinstance(x, (np.ndarray, torch.Tensor))
        assert x.shape[1] == self.ndof
        clamp_fn = torch.clamp if isinstance(x, torch.Tensor) else np.clip
        for i, (l, u) in enumerate(self.actuated_joints_limits):
            x[:, i] = clamp_fn(x[:, i], l, u)
        return x



    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Internal Functions                                             ---
    # ---                                                                                                            ---

    def _get_actuated_joint_child_names(self) -> List[str]:
        """Returns the names of the children of the actuated joints"""

        def get_child_name_from_joint_name(joint_name: str) -> str:
            for joint in self._end_effector_kinematic_chain:
                if joint.name == joint_name:
                    return joint.child
            raise ValueError(f"Could not find joint child of joint '{joint_name}'")

        return [get_child_name_from_joint_name(joint_name) for joint_name in self.actuated_joint_names]


    def __str__(self) -> str:
        s = "<Robot[{}] name:{}, ndofs:{}>".format(
            self.__class__.__name__,
            self.name,
            self.ndof,
        )
        return s

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Forward Kinematics                                             ---
    # ---                                                                                                            ---


    def _ensure_forward_kinematics_cache(self, device: torch.device, dtype: torch.dtype = DEFAULT_TORCH_DTYPE):
        if device not in self._parent_T_joint_cache:
            self._parent_T_joint_cache[device] = {}

        if self._end_effector_kinematic_chain[0].name not in self._parent_T_joint_cache[device]:
            for joint in self._end_effector_kinematic_chain:
                T = torch.diag_embed(torch.ones(1, 4, device=device, dtype=dtype))
                # TODO(@jstmn): Confirm that its faster to run `rpy_tuple_to_rotation_matrix` on the cpu and then send
                # to the gpu
                R = rpy_tuple_to_rotation_matrix(joint.origin_rpy, device=device)
                T[:, 0:3, 0:3] = R
                T[:, 0, 3] = joint.origin_xyz[0]
                T[:, 1, 3] = joint.origin_xyz[1]
                T[:, 2, 3] = joint.origin_xyz[2]
                self._parent_T_joint_cache[device][joint.name] = T.to(device)

            #
            if self._additional_link_name is not None:
                joint = self._additional_link_lca_joint
                T = torch.diag_embed(torch.ones(1, 4, device=device, dtype=dtype))
                # TODO(@jstmn): Confirm that its faster to run `rpy_tuple_to_rotation_matrix` on the cpu and then send
                # to the gpu
                R = rpy_tuple_to_rotation_matrix(joint.origin_rpy, device=device)
                T[:, 0:3, 0:3] = R
                T[:, 0, 3] = joint.origin_xyz[0]
                T[:, 1, 3] = joint.origin_xyz[1]
                T[:, 2, 3] = joint.origin_xyz[2]
                self._parent_T_joint_cache[device][joint.name] = T.to(device)

    def _batch_fk_iteration(self, joint: Joint, x_i: torch.Tensor, base_T_joint: torch.Tensor, out_device: str):
        assert joint.joint_type not in UNHANDLED_JOINT_TYPES, f"Joint type '{joint.joint_type}' is not implemented"
        assert joint.joint_type in (
            "revolute",
            "continuous",
            "prismatic",
            "fixed",
        ), f"Unknown joint type '{joint.joint_type}'"
        batch_size = x_i.shape[0]

        # translate + rotate joint frame by `origin_xyz`, `origin_rpy`
        parent_T_child_fixed = self._parent_T_joint_cache[out_device][joint.name].expand(
            batch_size, 4, 4
        )  # zero-copy expansion
        base_T_joint = base_T_joint.bmm(parent_T_child_fixed)

        if joint.joint_type in ("revolute", "continuous"):
            # rotate the joint frame about the `axis_xyz` axis by `x_i` radians
            rotation_axis = joint.axis_xyz
            # TODO: Implement a more efficient approach than converting to rotation matrices. work just with rpy?
            # or quaternions?
            joint_rotation = single_axis_angle_to_rotation_matrix(rotation_axis, x_i, out_device=out_device)
            assert joint_rotation.shape == (batch_size, 3, 3)
            T = torch.diag_embed(torch.ones(batch_size, 4, device=out_device))
            T[:, 0:3, 0:3] = joint_rotation
            return base_T_joint.bmm(T)

        if joint.joint_type == "prismatic":
            # Note: [..., None] is a trick to expand the x[:, x_i] tensor.
            translations = torch.tensor(joint.axis_xyz, device=out_device) * x_i[:, None]  # [batch x 3]
            assert translations.shape == (batch_size, 3)
            # TODO(@jstmn): consider making this more space efficient. create once and override?
            joint_fixed_T_joint = torch.diag_embed(torch.ones(batch_size, 4, device=out_device))
            joint_fixed_T_joint[:, 0:3, 3] = translations
            return base_T_joint.bmm(joint_fixed_T_joint)

        if joint.joint_type == "fixed":
            return base_T_joint

        raise RuntimeError(f"I shouldn't be here {joint.joint_type}")

    # TODO: Do FK starting at specific joint (like 'base_link') instead of the first joint.
    # TODO: Consider removing all cpu code from this function
    def forward_kinematics(
        self,
        x: torch.Tensor,
        out_device: Optional[str] = None,
        dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
        return_quaternion: bool = True,
        return_full_joint_fk: bool = False,
        return_full_link_fk: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Iterate through each joint in `_end_effector_kinematic_chain` and apply the joint's fixed transformation. If
        the joint is actuated then apply the angle from `x`.

        Args:
            x (torch.tensor): [N x ndof] tensor, storing the N configurations to calculate the robots FK for
            out_device (str): The device to save tensors to.
            return_quaternion (bool): Return format is [N x 7] where [:, 0:3] are xyz, and [:, 3:7] are quaternions.
                                        Otherwise return [N x 4 x 4] homogeneous transformation matrices

        Returns:
            if return_quaternion:
                torch.Tensor: [batch x 7] tensor of the end effector pose in quaternion format

            if return_full_joint_fk:
                torch.Tensor: [batch x ndof x 4 x 4] tensor of the pose of each actuated joint

            if return_full_link_fk:
                torch.Tensor: [batch x (s+1) (+ 1 if addl_link) x 4 x 4] tensor of the pose of each link in the end
                                                                            effectors kinematic chain. If there is an
                                                                            additional_link saved, the dimensionality of
                                                                            the second dimension will be (s+1) + 1

        """
        # TODO: return_full_link_fk needs to return additional_link poses as well
        assert self._batch_fk_enabled, f"BatchFK is disabled for '{self.name}'"

        batch_size = x.shape[0]
        _assert_is_joint_angle_matrix(x, self.ndof)
        if out_device is None:
            out_device = x.device

        self._ensure_forward_kinematics_cache(out_device, dtype=dtype)

        # Rotation from body/base link to joint along the joint chain
        base_T_joint = torch.diag_embed(torch.ones(batch_size, 4, device=out_device, dtype=dtype))
        base_T_joints = [base_T_joint]
        # Almost the same as base_T_joints except for fixed joints
        base_T_links = [base_T_joint]

        # Iterate through each joint in the joint chain
        # lca: lowest common ancestor. The index of the joint in the kinematic chain that is the lca of the
        # additional_link
        addl_link_lca_index = -1
        i = 0
        for joint in self._end_effector_kinematic_chain:
            # Check to see if this joint is the lca of the additional_link
            if self.additional_link is not None and joint.child == self._additional_link_lca_link:
                addl_link_lca_index = i

            i = min(
                i, self.ndof - 1
            )  # handles an edge case where a fixed joint after the final actuated joint causes an index error
            base_T_joint_new = self._batch_fk_iteration(joint, x[:, i], base_T_joint, out_device)
            base_T_links.append(base_T_joint_new)

            if joint.joint_type in ("revolute", "continuous"):
                base_T_joints.append(base_T_joint_new)
                i += 1
            if joint.joint_type == "prismatic":
                base_T_joints.append(base_T_joint_new)
                i += 1
            if joint.joint_type == "fixed":
                base_T_joints[-1] = base_T_joint_new
            base_T_joint = base_T_joint_new

        # Format output and return
        if self.additional_link is not None:
            assert addl_link_lca_index >= 0, "LCA link not found"
            # Note: the joint angles x[:, 0] will not be read, just need this to get the batch_size
            lca_link_T_addl_link = self._batch_fk_iteration(
                self._additional_link_lca_joint, x[:, 0], base_T_links[addl_link_lca_index + 1], out_device
            )
            base_T_links.append(lca_link_T_addl_link)

        if return_quaternion:
            quaternions = rotation_matrix_to_quaternion(base_T_joint[:, 0:3, 0:3])
            translations = base_T_joint[:, 0:3, 3]
            base_T_joint = torch.cat([translations, quaternions], dim=1)

        if return_full_joint_fk:
            ret = torch.stack(base_T_joints, dim=1)
            assert ret.shape == (batch_size, self.ndof + 1, 4, 4), f"ret.shape={ret.shape}"
            return ret

        if return_full_link_fk:
            ret = torch.stack(base_T_links, dim=1)
            n_links = len(self._end_effector_kinematic_chain) + 1
            if self.additional_link is not None:
                n_links += 1
            assert ret.shape == (batch_size, n_links, 4, 4), f"ret.shape={ret.shape} != ({batch_size}, {n_links}, 4, 4)"
            return ret

        return base_T_joint

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Inverse Kinematics                                             ---
    # ---                                                                                                            ---


    def jacobian(
        self,
        x: torch.Tensor,
        out_device: Optional[str] = None,
        dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
    ) -> torch.Tensor:
        """Return a batch of jacobian matrices for the given joint angle vectors. The rows are:

        0: roll (rx)
        1: pitch (ry)
        2: yaw (rz)
        3: x
        4: y
        5: z

        Args:
            x (torch.Tensor): The joint angle to the jacobian with respect to

        Returns:
            torch.Tensor: a [6 x ndof] matrix, where the top 3 rows are the orientation derivatives, and the bottom
                            three rows are the positional derivatives.
        """

        batch_size = x.shape[0]
        _assert_is_joint_angle_matrix(x, self.ndof)
        if out_device is None:
            out_device = x.device

        self._ensure_forward_kinematics_cache(out_device, dtype=dtype)

        # Rotation from body/base link to joint along the joint chain
        base_T_joints = self.forward_kinematics(x, return_full_joint_fk=True, out_device=out_device, dtype=dtype)
        base_T_joints = base_T_joints[:, 1:, :, :]  # remove the base link

        # Compute jacobian
        J = torch.zeros((batch_size, 6, self.ndof), device=out_device, dtype=dtype)
        x_i = 0
        for joint in self._end_effector_kinematic_chain:
            assert joint.joint_type not in UNHANDLED_JOINT_TYPES, f"Joint type '{joint.joint_type}' is not implemented"
            if joint.joint_type in ("revolute", "continuous"):
                axis = (
                    torch.tensor(joint.axis_xyz, device=out_device, dtype=dtype).unsqueeze(1).expand(batch_size, 3, 1)
                )
                J[:, :3, x_i] = base_T_joints[:, x_i, :3, :3].bmm(axis).squeeze(2)
                d = base_T_joints[:, -1, :3, 3] - base_T_joints[:, x_i, :3, 3]
                world_axis = base_T_joints[:, x_i, :3, :3].bmm(axis).squeeze(2)
                J[:, 3:6, x_i] = torch.cross(world_axis, d, dim=1)

                x_i += 1

            elif joint.joint_type == "prismatic":
                J[:, 3, x_i], J[:, 4, x_i], J[:, 5, x_i] = joint.axis_xyz

                x_i += 1

        return J

    def inverse_kinematics_step_levenburg_marquardt(
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

        lfs_A = torch.bmm(J_batch_T, J_batch) + lambd * eye  # [n ndof ndof]
        rhs_B = torch.bmm(J_batch_T, pose_errors)  # [n ndof 1]
        delta_x = torch.linalg.solve(lfs_A, rhs_B)  # [n ndof 1]

        if alphas is not None:
            assert alphas.shape == (n, 1)
            xs_updated = xs_current + alphas * torch.squeeze(delta_x)
        else:
            xs_updated = xs_current + alpha * torch.squeeze(delta_x)

        if clamp_to_joint_limits:
            return self.clamp_to_joint_limits(xs_updated)
        return xs_updated

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
        assert (
            xs_current.device == target_poses.device
        ), f"xs_current and target_poses must be on the same device (got {xs_current.device} and {target_poses.device})"
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
        assert (
            current_poses.shape == target_poses.shape
        ), f"current_poses.shape != target_poses.shape ({current_poses.shape} != {target_poses.shape})"

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
        assert (
            torch.isnan(pose_errors).sum() == 0
        ), f"pose_errors contains NaNs ({torch.isnan(pose_errors).sum()} of {pose_errors.numel()})"

        # tensor dimensions: [batch x ndofs x 6] * [batch x 6 x 1] = [batch x ndofs x 1]
        delta_x = J_pinv @ pose_errors
        xs_updated = xs_current + alpha * delta_x[:, :, 0]

        assert torch.isnan(xs_updated).sum() == 0, "xs_updated contains NaNs"
        assert xs_current.device == xs_updated.device
        xs_updated = self.clamp_to_joint_limits(xs_updated)
        return xs_updated


    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                            Collision Detection                                             ---
    # ---                                                                                                            ---

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
        TODO
        raise NotImplementedError("Implement self_collision_distances_jacobian")


    def self_collision_distances_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the jacobian of the self collision distance function with respect to the joint angles.

        Args:
            x (torch.Tensor): [n x ndofs] joint angle vectors

        Returns:
            torch.Tensor: [n x n_pairs x ndofs] jacobian
        """
        TODO
        raise NotImplementedError("Implement self_collision_distances_jacobian")


def forward_kinematics_kinpy(robot: Robot, x: np.array) -> np.array:
    """
    Returns the pose of the end effector for each joint parameter setting in x
    """
    _assert_is_joint_angle_matrix(x, robot.ndof)

    import kinpy as kp

    with open(robot.urdf_filepath) as f:
        kinpy_fk_chain = kp.build_chain_from_urdf(f.read().encode("utf-8"))

    n = x.shape[0]
    y = np.zeros((n, 7))
    zero_transform = kp.transform.Transform()
    fk_dict = {}
    for joint_name in robot.actuated_joint_names:
        fk_dict[joint_name] = 0.0

    def get_fk_dict(xs):
        for i in range(robot.ndof):
            fk_dict[robot.actuated_joint_names[i]] = xs[i]
        return fk_dict

    for i in range(n):
        th = get_fk_dict(x[i])
        transform = kinpy_fk_chain.forward_kinematics(th, world=zero_transform)[robot.end_effector_link_name]
        y[i, 0:3] = transform.pos
        y[i, 3:] = transform.rot
    return y
