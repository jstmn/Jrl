from typing import List, Tuple, Optional, Union, Dict
from functools import cached_property

from more_itertools import locate
import torch
import numpy as np
import klampt
from klampt import IKSolver, WorldModel
from klampt.model import ik
from klampt.math import so3
from klampt import robotsim
from klampt.model.collide import WorldCollider
import tqdm

from jrl.math_utils import (
    rpy_tuple_to_rotation_matrix,
    single_axis_angle_to_rotation_matrix,
    quaternion_inverse,
    quaternion_product,
    quaternion_to_rpy,
    rotation_matrix_to_quaternion,
    quaternion_norm,
    geodesic_distance_between_quaternions,
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
from jrl.utils import to_torch
from jrl.geometry import capsule_capsule_distance_batch, capsule_cuboid_distance_batch, _get_capsule_axis_endpoints


def _assert_is_2d(x: Union[torch.Tensor, np.ndarray]):
    assert len(x.shape) == 2, f"Expected x to be a 2D array but got {x.shape}"
    assert isinstance(x, (torch.Tensor, np.ndarray)), f"Expected x to be a torch.Tensor or np.ndarray but got {type(x)}"


def _assert_is_pose_matrix(poses: Union[torch.Tensor, np.ndarray]):
    _assert_is_2d(poses)
    assert poses.shape[1] == 7, f"Expected poses matrix to be [n x 7] but got {poses.shape}"
    norms = quaternion_norm(poses[:, 3:7])
    assert max(norms) < 1.01 and min(norms) > 0.99, "quaternion(s) are not unit quaternion(s)"


def _assert_is_joint_angle_matrix(xs: Union[torch.Tensor, np.ndarray], ndof: int):
    _assert_is_2d(xs)
    assert xs.shape[1] == ndof, f"Expected matrix to be [n x ndof] ([{xs.shape[0]} x {ndof}]) but got {xs.shape}"


def _assert_is_np(x: np.ndarray, variable_name: str = "input"):
    assert isinstance(x, np.ndarray), f"Expected {variable_name} to be a numpy array but got {type(x)}"


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

        # Initialize klampt
        # Note: Need to save `_klampt_world_model` as a member variable otherwise you'll be doomed to get a segfault
        self._klampt_world_model = WorldModel()
        # TODO: Consider finding a better way to fix the mesh filepath issue.
        self._urdf_filepath_absolute = get_urdf_filepath_w_filenames_updated(self._urdf_filepath)
        self._klampt_world_model.loadRobot(self._urdf_filepath_absolute)  # TODO: suppress output of loadRobot call
        assert (
            self._klampt_world_model.numRobots()
        ), f"There should be one robot loaded (found {self._klampt_world_model.numRobots()}). Is the urdf well formed?"
        self._klampt_robot: robotsim.RobotModel = self._klampt_world_model.robot(0)
        self._ignored_collision_pairs_formatted = [
            (self._klampt_robot.link(link1_name), self._klampt_robot.link(link2_name))
            for link1_name, link2_name in ignored_collision_pairs
        ]
        self._klampt_collision_checker = WorldCollider(
            self._klampt_world_model, ignore=self._ignored_collision_pairs_formatted
        )
        self._ignored_collision_pairs = ignored_collision_pairs + [(l2, l1) for l1, l2 in ignored_collision_pairs]
        self._klampt_ee_link: robotsim.RobotModelLink = self._klampt_robot.link(self._end_effector_link_name)
        self._klampt_config_dim = len(self._klampt_robot.getConfig())
        self._klampt_driver_vec_dim = self._klampt_robot.numDrivers()
        self._klampt_active_dofs = self._get_klampt_active_dofs()
        self._klampt_active_driver_idxs = self._get_klampt_active_driver_idxs()

        if verbose:
            print("\n----------- Robot specs")
            print(f"name: {self.name}")
            print("klampt config size: ", self._klampt_config_dim)
            print("klampt drivers size:", self._klampt_driver_vec_dim)
            print("joints:")
            sjld = 0
            for i, (joint_name, (l, u)) in enumerate(zip(self.actuated_joint_names, self.actuated_joints_limits)):
                print(f"  {i} {joint_name}:\t{round(l, 3)},\t{round(u, 3)}")
                sjld += u - l
            print(f"sum joint range: {round(sjld, 4)} rads")
            print("-----------\n")

    def make_new_collision_checker(self):
        self._klampt_collision_checker = WorldCollider(
            self._klampt_world_model, ignore=self._ignored_collision_pairs_formatted
        )

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
    def klampt_world_model(self) -> WorldModel:
        return self._klampt_world_model

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
                samples_i = self.sample_joint_angles(internal_batch_size, joint_limit_eps=joint_limit_eps)
                counter0_i = counter

                for i in range(samples_i.shape[0]):
                    sample = samples_i[i]
                    if only_non_self_colliding and self.config_self_collides(sample):
                        continue

                    pose = self.forward_kinematics_klampt(sample[None, :])
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

    def set_klampt_robot_config(self, x: np.ndarray):
        """Set the internal klampt robots config with the given joint angle vector"""
        _assert_is_np(x)
        assert x.shape == (self.ndof,), f"Expected x to be of shape ({self.ndof},) but got {x.shape}"
        q = self._x_to_qs(np.resize(x, (1, self.ndof)))[0]
        self._klampt_robot.setConfig(q)

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

    def config_self_collides(self, x: PT_NP_TYPE, verbose: bool = False) -> bool:
        """Returns True if the given joint angle vector causes the robot to self collide

        Args:
            x (PT_NP_TYPE): [ndofs] tensor of joint angles
        """
        assert isinstance(x, (np.ndarray, torch.Tensor))
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.set_klampt_robot_config(x)
        collisions = self._klampt_collision_checker.robotSelfCollisions(self._klampt_robot)
        if verbose:
            is_first = True
            for i, (link1, link2) in enumerate(collisions):
                if is_first:
                    print(f"\nCollisions")
                    is_first = False
                print(f"collision {i}: {link1.getName()} {link2.getName()}")
        for _ in collisions:
            return True
        return False

    def config_collides_with_env(self, x: PT_NP_TYPE, box: klampt.Geometry3D, return_detailed: bool = False) -> bool:
        """Returns True if the given joint angle vector causes the robot to self collide

        Args:
            x (PT_NP_TYPE): [ndofs] tensor of joint angles
        """
        assert isinstance(x, (np.ndarray, torch.Tensor))
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.set_klampt_robot_config(x)
        collisions = self._klampt_collision_checker.robotObjectCollisions(self._klampt_robot, box)

        if return_detailed:
            collisions_list = list(collisions)
            return collisions_list

        # Not sure if collisions lazily evaluated or not, so we need to iterate over instead of calling list(collisions)
        # (RobotModelLink, RigidObjectModel) (for link, rigid_object in collisions:)
        for _ in collisions:
            return True
        return False

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

    def _get_klampt_active_dofs(self) -> List[int]:
        """Hack: We need to know which indexes of the klampt q vector are from active joints.

        Returns:
            List[int]: The indexes of the klampt configuration vector which correspond to the user specified active
                        joints
        """
        all_drivers = [self._klampt_robot.driver(i) for i in range(self._klampt_robot.numDrivers())]
        actuated_joint_child_names = self._get_actuated_joint_child_names()
        driver_vec_tester = [1000 if (driver.getName() in actuated_joint_child_names) else -1 for driver in all_drivers]
        q_test_result = self._klampt_robot.configFromDrivers(driver_vec_tester)
        q_active_joint_idxs = list(locate(q_test_result, lambda x: x == 1000))
        assert len(q_active_joint_idxs) == self.ndof, (
            f"Error - the number of active drivers in the klampt config != ndof ({len(q_active_joint_idxs)} !="
            f" {self.ndof})"
        )
        return q_active_joint_idxs

    def _get_klampt_active_driver_idxs(self) -> List[int]:
        """We need to know which indexes of the klampt driver vector are from user specified active joints.

        Returns:
            List[int]: The indexes of the klampt driver vector which correspond to the(user specified) active joints
        """

        # Get the names of all the child links for each active joint.
        # Note: driver.getName() returns the joints' child link for some god awful reason. See L1161 in Robot.cpp
        # (https://github.com/krishauser/Klampt/blob/master/Cpp/Modeling/Robot.cpp#L1161)
        actuated_joint_child_names = self._get_actuated_joint_child_names()
        all_drivers = [self._klampt_robot.driver(i) for i in range(self._klampt_robot.numDrivers())]
        driver_vec_tester = [1 if (driver.getName() in actuated_joint_child_names) else -1 for driver in all_drivers]
        active_driver_idxs = list(locate(driver_vec_tester, lambda x: x == 1))

        assert (
            len(active_driver_idxs) == self.ndof
        ), f"Error - the number of active drivers != ndof ({len(active_driver_idxs)} != {self.ndof})"
        return active_driver_idxs

    # TODO(@jstm): Consider changing this to take (batch x ndofs)
    def _driver_vec_from_x(self, x: np.ndarray) -> List[float]:
        """Format a joint angle vector into a klampt driver vector. Non user specified joints will have a value of 0.

        Args:
            x (np.ndarray): (self.ndof,) joint angle vector

        Returns:
            List[float]: A list with the joint angle vector formatted in the klampt driver format. Note that klampt
                            needs a list of floats when recieving a driver vector.
        """
        assert x.size == self.ndof, f"x doesn't have {self.ndof} (ndof) elements ({self.ndof} != {x.size})"
        assert x.shape == (self.ndof,), f"x.shape must be (ndof,) - ({(self.ndof,)}) != {x.shape}"
        x = x.tolist()

        # return x as a list if there are no additional active joints in the urdf
        if len(x) == self._klampt_driver_vec_dim:
            return x

        # TODO(@jstm): Consider a non iterative implementation for this
        driver_vec = [0.0] * self._klampt_driver_vec_dim

        j = 0
        for i in self._klampt_active_driver_idxs:
            driver_vec[i] = x[j]
            j += 1

        return driver_vec

    # TODO(@jstm): Consider changing this to take (batch x driver_vec_dim)
    def _x_from_driver_vec(self, driver_vec: List[float]) -> List[float]:
        """Remove the non relevant joints from the klampt driver vector."""
        if len(driver_vec) == self.ndof:
            return driver_vec
        # TODO(@jstm): Consider a non iterative implementation for this
        return [driver_vec[i] for i in self._klampt_active_driver_idxs]

    def _x_to_qs(self, x: np.ndarray) -> List[List[float]]:
        """Return a list of klampt configurations (qs) from an array of joint angles (x)

        Args:
            x: (n x ndof) array of joint angle settings

        Returns:
            A list of configurations representing the robots state in klampt
        """
        _assert_is_np(x)
        _assert_is_joint_angle_matrix(x, self.ndof)

        n = x.shape[0]
        qs = []
        for i in range(n):
            driver_vec = self._driver_vec_from_x(x[i])
            qs.append(self._klampt_robot.configFromDrivers(driver_vec))
        return qs

    def _qs_to_x(self, qs: List[List[float]]) -> np.array:
        """Calculate joint angle values (x) from klampt configurations (qs)"""
        res = np.zeros((len(qs), self.ndof))
        for idx, q in enumerate(qs):
            driver_vec = self._klampt_robot.configToDrivers(q)
            res[idx, :] = self._x_from_driver_vec(driver_vec)
        return res

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

    def forward_kinematics_klampt(self, x: np.array, link_name: Optional[str] = None) -> np.array:
        """Forward kinematics using the klampt library"""
        robot_configs = self._x_to_qs(x)
        dim_y = 7
        n = len(robot_configs)
        y = np.zeros((n, dim_y))

        if link_name is None:
            link = self._klampt_ee_link
        else:
            link = self._klampt_robot.link(link_name)

        for i in range(n):
            q = robot_configs[i]
            self._klampt_robot.setConfig(q)
            R, t = link.getTransform()
            y[i, 0:3] = np.array(t)
            y[i, 3:] = np.array(so3.quaternion(R))
        return y

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

    def jacobian_np(self, x: np.ndarray) -> np.ndarray:
        """Return the jacobian of the end effector link with respect to the joint angles x.

        Note: per Klamp't, the format for the returned orientation:
            > The x axis is "roll", y is "pitch", and z is "yaw"

        Args:
            x (np.ndarray): The joint angle to the jacobian with respect to

        Returns:
            np.ndarray: a [6 x ndof] matrix, where the top 3 rows are the orientation derivatives, and the bottom three
                        rows are the positional derivatives.
        """
        self.set_klampt_robot_config(x)
        J_full = self._klampt_ee_link.getJacobian([0, 0, 0])  # [6 x klampt_config_dimension]
        J = J_full[:, self._klampt_active_dofs]  # see https://stackoverflow.com/a/8386737/5191069
        return J

    def jacobian_batch_np(self, xs: np.ndarray) -> np.ndarray:
        """Return a batch of jacobian matrices for the given joint angle vectors. See 'jacobian_np()' for details on
        the jacobian

        Args:
            xs (np.ndarray): [batch x ndof] matrix of joint angle vectors

        Returns:
            np.ndarray: _description_
        """
        _assert_is_joint_angle_matrix(xs, self.ndof)
        n = xs.shape[0]
        Js = np.zeros((n, 6, self.ndof))
        for i in range(n):
            Js[i] = self.jacobian_np(xs[i])
        return Js

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

    def inverse_kinematics_klampt(
        self,
        pose: np.array,
        seed: Optional[np.ndarray] = None,
        positional_tolerance: float = 1e-3,
        n_tries: int = 50,
        verbosity: int = 0,
    ) -> Optional[np.array]:
        """Run klampts inverse kinematics solver with the given pose

        Note: If the solver fails to find a solution with the provided seed, it will rerun with a random seed

        Per http://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/Manual-IK.html#ik-solver:
        'To use the solver properly, you must understand how the solver uses the RobotModel:
            First, the current configuration of the robot is the seed configuration to the solver.
            Second, the robot's joint limits are used as the defaults.
            Third, the solved configuration is stored in the RobotModel's current configuration.'

        Args:
            pose (np.array): The target pose to solve for
            seed (Optional[np.ndarray], optional): A seed to initialize the optimization with. Defaults to None.
            verbosity (int): Set the verbosity of the function. 0: only fatal errors are printed. Defaults to 0.
        """
        assert len(pose.shape) == 1
        assert pose.size == 7, f"Error, pose is {pose.shape}, should be [{7}]"
        if seed is not None:
            _assert_is_np(seed, variable_name="seed")
            assert len(seed.shape) == 1, f"Seed must be a 1D array (currently: {seed.shape})"
            assert seed.size == self.ndof
            seed_q = self._x_to_qs(seed.reshape((1, self.ndof)))[0]

        max_iterations = 150
        R = so3.from_quaternion(pose[3 : 3 + 4])
        obj = ik.objective(self._klampt_ee_link, t=pose[0:3].tolist(), R=R)

        for _ in range(n_tries):
            solver = IKSolver(self._klampt_robot)
            solver.add(obj)
            solver.setActiveDofs(self._klampt_active_dofs)
            solver.setMaxIters(max_iterations)
            # TODO(@jstmn): What does 'tolarance' mean for klampt? Positional error? Positional error + orientation
            # error?
            solver.setTolerance(positional_tolerance)

            # `sampleInitial()` needs to come after `setActiveDofs()`, otherwise x,y,z,r,p,y of the robot will
            # be randomly set aswell <(*<_*)>
            if seed is None:
                solver.sampleInitial()
            else:
                # solver.setBiasConfig(seed_q)
                self._klampt_robot.setConfig(seed_q)

            res = solver.solve()
            if not res:
                if verbosity > 0:
                    print(
                        "  inverse_kinematics_klampt() IK failed after",
                        solver.lastSolveIters(),
                        "optimization steps, retrying (non fatal)",
                    )

                # Rerun the solver with a random seed
                if seed is not None:
                    return self.inverse_kinematics_klampt(
                        pose,
                        seed=None,
                        positional_tolerance=positional_tolerance,
                        verbosity=verbosity,
                    )

                continue

            if verbosity > 1:
                print("Solved in", solver.lastSolveIters(), "iterations")
                residual = solver.getResidual()
                print("Residual:", residual, " - L2 error:", np.linalg.norm(residual[0:3]))

            return self._qs_to_x([self._klampt_robot.getConfig()])

        if verbosity > 0:
            print(
                "inverse_kinematics_klampt() - Failed to find IK solution after",
                n_tries,
                "optimization attempts",
            )
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                            Collision Detection                                             ---
    # ---                                                                                                            ---

    def get_capsule_axis_endpoints(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the endpoints of the capsules attached to each link, in the robots base frame for a single joint
        angle vector

        Args:
            x (torch.Tensor): [1 x ndofs] joint angle vector

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (endpoints1, endpoints2, radii) where endpoints1 and
            endpoints2 are the endpoints of the capsules in the robots base frame, and radii are the radii of the capsules
        """
        assert x.shape[0] == 1
        assert x.shape[1] == self.ndof
        n_capsules = len(self._collision_capsules_by_link)
        base_T_links = self.forward_kinematics(x, return_full_link_fk=True, out_device=x.device, dtype=x.dtype).view(
            n_capsules, 4, 4
        )
        capsule_params = torch.cat([v.view(1, 7) for v in self._collision_capsules_by_link.values()], axis=0).to(
            x.device
        )
        # Capsules are defined by two points in local frame, and a radius. The memory layout is
        # [nx7]: [x1, y1, z1, x2, y2, z2, r1].
        caps_radius = capsule_params[:, 6]
        caps_p1, caps_p2 = _get_capsule_axis_endpoints(capsule_params, base_T_links)
        return caps_p1, caps_p2, caps_radius

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
        """Returns the jacobian of the self collision distance function with
        respect to the joint angles.

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
        """Returns the distance between the robot collision capsules and the
        environment cuboid obstacle for each joint angle vector in x.

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
