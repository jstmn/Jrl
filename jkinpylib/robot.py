from typing import List, Tuple, Optional, Union, Dict
from time import time
from functools import cached_property

from more_itertools import locate
import torch
import numpy as np
from klampt import IKSolver, WorldModel
from klampt.model import ik
from klampt.math import so3
from klampt import robotsim
from klampt.model.collide import WorldCollider
import tqdm
from copy import deepcopy

from jkinpylib.conversions import (
    rpy_tuple_to_rotation_matrix,
    single_axis_angle_to_rotation_matrix,
    quaternion_inverse,
    quaternion_product,
    quaternion_to_rpy,
    rotation_matrix_to_quaternion,
    quaternion_norm,
    geodesic_distance_between_quaternions,
    DEFAULT_TORCH_DTYPE,
    PT_NP_TYPE,
)
from jkinpylib.config import DEVICE, ACCELERATOR_AVAILABLE
from jkinpylib.urdf_utils import (
    Joint,
    get_joint_chain,
    get_urdf_filepath_w_filenames_updated,
    UNHANDLED_JOINT_TYPES,
)
from jkinpylib.geometry import capsule_capsule_distance_batch, capsule_cuboid_distance_batch


def _assert_is_2d(x: Union[torch.Tensor, np.ndarray]):
    assert len(x.shape) == 2, f"Expected x to be a 2D array but got {x.shape}"
    assert isinstance(x, (torch.Tensor, np.ndarray)), f"Expected x to be a torch.Tensor or np.ndarray but got {type(x)}"


def _assert_is_pose_matrix(poses: Union[torch.Tensor, np.ndarray]):
    _assert_is_2d(poses)
    assert poses.shape[1] == 7, f"Expected poses matrix to be [n x 7] but got {poses.shape}"
    norms = quaternion_norm(poses[:, 3:7])
    assert max(norms) < 1.01 and min(norms) > 0.99, "quaternion(s) are not unit quaternion(s)"


def _assert_is_joint_angle_matrix(xs: Union[torch.Tensor, np.ndarray], n_dofs: int):
    _assert_is_2d(xs)
    assert xs.shape[1] == n_dofs, f"Expected matrix to be [n x n_dofs] ([{xs.shape[0]} x {n_dofs}]) but got {xs.shape}"


def _assert_is_np(x: np.ndarray, variable_name: str = "input"):
    assert isinstance(x, np.ndarray), f"Expected {variable_name} to be a numpy array but got {type(x)}"


def _generate_self_collision_pairs(
    collision_capsules_by_link: Dict[str, torch.Tensor],
    joint_chain: List[Joint],
    ignored_collision_pairs: List[Tuple[str, str]] = [],
):
    """
    Generate collision pairs from collision capsules and joint chain. Adjacent
    links in the joint chain are allowed to collide.

    Returns capsules and idx0, idx1, such that
        capsules[idx0[i]] and capsules[idx1[i]]
    must be checked for collision.

    Args:
        joint_chain (List[Joint]): Joint chain of the robot.
        ignored_collision_pairs (List[Tuple[str, str]], optional): List of collision pairs to ignore. Defaults to [].

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: capsules, idx0, idx1
    """
    ignored_collision_set = set(tuple(sorted(pair)) for pair in ignored_collision_pairs)
    for link_name, capsule in collision_capsules_by_link.items():
        if capsule is None:
            for other_link_name in collision_capsules_by_link.keys():
                ignored_collision_set.add(tuple(sorted((link_name, other_link_name))))
        collision_capsules_by_link[link_name] = torch.tensor([0, 0, 0, 0, 0, 0.01, 0.001], device=DEVICE, dtype=DEFAULT_TORCH_DTYPE)

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


# TODO: Add base link
class Robot:
    def __init__(
        self,
        name: str,
        urdf_filepath: str,
        active_joints: List[str],
        base_link: str,
        end_effector_link_name: str,
        ignored_collision_pairs: List[Tuple[str, str]],
        positional_repeatability_mm: float,
        rotational_repeatability_deg: float,
        collision_capsules_by_link: Optional[Dict[str, torch.Tensor]] = None,
        batch_fk_enabled: bool = True,
        verbose: bool = False,
    ):
        """TODO

        Args:
            name (str): _description_
            base_link (str): _description_
            urdf_filepath (str): _description_
            active_joints (List[str]): The name of the actuated joints in the kinematic chain that is being represented.
                                        These joints must be along the link chain from the 'base_link' to the
                                        'end_effector_link_name'. Note that all non-fixed joints in this chain that are
                                        not in 'active_joints' will be ignored (by being changed to fixed joints).
            end_effector_link_name (str): _description_
            ignored_collision_pairs (List[Tuple[str, str]]): _description_
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
        self._collision_capsules_by_link = collision_capsules_by_link
        self._batch_fk_enabled = batch_fk_enabled
        self._positional_repeatability_mm = positional_repeatability_mm
        self._rotational_repeatability_deg = rotational_repeatability_deg

        # Note: `_joint_chain`, `_actuated_joint_limits`, `_actuated_joint_names` only includes the joints that were
        # specified by the subclass. It does not include all actuated joints in the urdf
        self._joint_chain = get_joint_chain(
            self._urdf_filepath,
            active_joints,
            self._base_link,
            self._end_effector_link_name,
        )
        self._actuated_joint_limits = [joint.limits for joint in self._joint_chain if joint.is_actuated]
        self._actuated_joint_names = [joint.name for joint in self._joint_chain if joint.is_actuated]
        self._actuated_joint_velocity_limits = [
            joint.velocity_limit for joint in self._joint_chain if joint.is_actuated
        ]
        assert len(active_joints) == self.n_dofs, (
            f"Error - the number of active joints ({len(active_joints)}) does not match the degrees of freedom"
            f" ({self.n_dofs})."
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
                self._joint_chain,
                ignored_collision_pairs,
            )

        # Create and fill cache of fixed rotations between links.
        self._parent_T_joint_cache = {}
        if self._batch_fk_enabled:
            self.forward_kinematics_batch(
                torch.tensor(
                    self.sample_joint_angles(1050),
                    device=DEVICE,
                    dtype=DEFAULT_TORCH_DTYPE,
                )
            )
        # self.forward_kinematics_batch(
        #     torch.tensor(self.sample_joint_angles(1000), device="cpu", dtype=DEFAULT_TORCH_DTYPE), out_device="cpu"
        # )

        # Initialize klampt
        # Note: Need to save `_klampt_world_model` as a member variable otherwise you'll be doomed to get a segfault
        self._klampt_world_model = WorldModel()

        # TODO: Consider finding a better way to fix the mesh filepath issue. This feels pretty hacky. But hey, it works
        # <insert-shrug-emoji>
        self._urdf_filepath_absolute = get_urdf_filepath_w_filenames_updated(self._urdf_filepath)
        self._klampt_world_model.loadRobot(self._urdf_filepath_absolute)  # TODO: supress output of loadRobot call
        assert (
            self._klampt_world_model.numRobots()
        ), f"There should be one robot loaded (found {self._klampt_world_model.numRobots()}). Is the urdf well formed?"
        self._klampt_robot: robotsim.RobotModel = self._klampt_world_model.robot(0)
        ignored_collision_pairs_formatted = [
            (self._klampt_robot.link(link1_name), self._klampt_robot.link(link2_name))
            for link1_name, link2_name in ignored_collision_pairs
        ]
        self._klampt_collision_checker = WorldCollider(
            self._klampt_world_model, ignore=ignored_collision_pairs_formatted
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

    @cached_property
    def n_dofs(self) -> int:
        return sum([1 for joint in self._joint_chain if joint.is_actuated])

    @property
    def n_collision_pairs(self) -> int:
        """Return the number of capsule-capsule collision pairs of the robot"""
        assert self._collision_capsules is not None, f"collision_capsules not defined for '{self.name}'"
        return self._collision_idx0.numel()

    @property
    def n_joints_in_kinematic_chain(self) -> int:
        """The number of joints in the path from the base_link to the end effector"""
        return len(self._joint_chain)

    @property
    def actuated_joint_names(self) -> List[str]:
        return self._actuated_joint_names

    @cached_property
    def actuated_joint_types(self) -> List[str]:
        return [joint.joint_type for joint in self._joint_chain if joint.is_actuated]

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
        for joint in self._joint_chain:
            if joint.is_actuated and (joint.joint_type == "revolute" or joint.joint_type == "continuous"):
                vals.append(joint.velocity_limit * 180 / np.pi)
            elif joint.is_actuated and (joint.joint_type == "prismatic"):
                vals.append(joint.velocity_limit)

        assert len(vals) == self.n_dofs, f"Error, only {len(vals)} in vals, but {self.n_dofs} degrees of freedom"
        return vals

    @property
    def klampt_world_model(self) -> WorldModel:
        return self._klampt_world_model

    @property
    def positional_repeatability_mm(self) -> float:
        return self._positional_repeatability_mm

    @property
    def rotational_repeatability_deg(self) -> float:
        return self._rotational_repeatability_deg

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             External Functions                                             ---
    # ---                                                                                                            ---

    def split_configs_to_revolute_and_prismatic(self, configs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Returns the values for the values from the revolute, and prismatic joints separately"""
        assert configs.shape[1] == self.n_dofs
        joint_types = self.actuated_joint_types
        revolute_idxs = [i for i in range(self.n_dofs) if joint_types[i] in {"revolute", "continuous"}]
        prismatic_idxs = [i for i in range(self.n_dofs) if joint_types[i] == "prismatic"]
        assert len(revolute_idxs) + len(prismatic_idxs) == self.n_dofs
        return configs[:, revolute_idxs], configs[:, prismatic_idxs]

    def sample_joint_angles(self, n: int, joint_limit_eps: float = 1e-6) -> np.ndarray:
        """Returns a [N x ndof] matrix of randomly drawn joint angle vectors. The joint angles are sampled from the
        range [lower+joint_limit_eps, upper-joint_limit_eps]
        """
        angs = np.random.rand(n, self.n_dofs)  # between [0, 1)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a [N x ndof] matrix of randomly drawn joint angle vectors with matching end effector poses."""
        samples = np.zeros((n, self.n_dofs))
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
        assert x.shape == (self.n_dofs,), f"Expected x to be of shape ({self.n_dofs},) but got {x.shape}"
        q = self._x_to_qs(np.resize(x, (1, self.n_dofs)))[0]
        self._klampt_robot.setConfig(q)

    def clamp_to_joint_limits(self, x: PT_NP_TYPE):
        """Clamp the given joint angle vectors to the joint limits

        Args:
            x (PT_NP_TYPE): [batch x ndofs] tensor of joint angles
        """
        assert isinstance(x, (np.ndarray, torch.Tensor))
        assert x.shape[1] == self.n_dofs
        clamp_fn = torch.clamp if isinstance(x, torch.Tensor) else np.clip
        for i, (l, u) in enumerate(self.actuated_joints_limits):
            x[:, i] = clamp_fn(x[:, i], l, u)
        return x

    def joint_angles_all_in_joint_limits(self, x: PT_NP_TYPE) -> bool:
        """Return whether all joint angles are within joint limits

        Args:
            x (PT_NP_TYPE): [batch x ndofs] tensor of joint angles
        """
        assert isinstance(x, (np.ndarray, torch.Tensor))
        assert x.shape[1] == self.n_dofs
        for i, (lower, upper) in enumerate(self.actuated_joints_limits):
            if x[:, i].min() < lower:
                return False
            if x[:, i].max() > upper:
                return False
        return True

    def config_self_collides(self, x: PT_NP_TYPE, debug_mode: bool = False) -> bool:
        """Returns True if the given joint angle vector causes the robot to self collide

        Args:
            x (PT_NP_TYPE): [ndofs] tensor of joint angles
        """
        assert isinstance(x, (np.ndarray, torch.Tensor))
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.set_klampt_robot_config(x)
        collisions = self._klampt_collision_checker.robotSelfCollisions(self._klampt_robot)
        if debug_mode:
            is_collision = False
        for link1, link2 in collisions:
            if debug_mode:
                print(f"Collision between {link1.getName()} and {link2.getName()}")
                is_collision = True
            else:
                return True
        if debug_mode and is_collision:
            return True
        return False

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Internal Functions                                             ---
    # ---                                                                                                            ---

    def _get_actuated_joint_child_names(self) -> List[str]:
        """Returns the names of the children of the actuated joints"""

        def get_child_name_from_joint_name(joint_name: str) -> str:
            for joint in self._joint_chain:
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
        assert len(q_active_joint_idxs) == self.n_dofs, (
            f"Error - the number of active drivers in the klampt config != n_dofs ({len(q_active_joint_idxs)} !="
            f" {self.n_dofs})"
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
            len(active_driver_idxs) == self.n_dofs
        ), f"Error - the number of active drivers != n_dofs ({len(active_driver_idxs)} != {self.n_dofs})"
        return active_driver_idxs

    # TODO(@jstm): Consider changing this to take (batch x ndofs)
    def _driver_vec_from_x(self, x: np.ndarray) -> List[float]:
        """Format a joint angle vector into a klampt driver vector. Non user specified joints will have a value of 0.

        Args:
            x (np.ndarray): (self.n_dofs,) joint angle vector

        Returns:
            List[float]: A list with the joint angle vector formatted in the klampt driver format. Note that klampt
                            needs a list of floats when recieving a driver vector.
        """
        assert x.size == self.n_dofs, f"x doesn't have {self.n_dofs} (n_dofs) elements ({self.n_dofs} != {x.size})"
        assert x.shape == (self.n_dofs,), f"x.shape must be (n_dofs,) - ({(self.n_dofs,)}) != {x.shape}"
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
        if len(driver_vec) == self.n_dofs:
            return driver_vec
        # TODO(@jstm): Consider a non iterative implementation for this
        return [driver_vec[i] for i in self._klampt_active_driver_idxs]

    def _x_to_qs(self, x: np.ndarray) -> List[List[float]]:
        """Return a list of klampt configurations (qs) from an array of joint angles (x)

        Args:
            x: (n x n_dofs) array of joint angle settings

        Returns:
            A list of configurations representing the robots state in klampt
        """
        _assert_is_np(x)
        _assert_is_joint_angle_matrix(x, self.n_dofs)

        n = x.shape[0]
        qs = []
        for i in range(n):
            driver_vec = self._driver_vec_from_x(x[i])
            qs.append(self._klampt_robot.configFromDrivers(driver_vec))
        return qs

    def _qs_to_x(self, qs: List[List[float]]) -> np.array:
        """Calculate joint angle values (x) from klampt configurations (qs)"""
        res = np.zeros((len(qs), self.n_dofs))
        for idx, q in enumerate(qs):
            driver_vec = self._klampt_robot.configToDrivers(q)
            res[idx, :] = self._x_from_driver_vec(driver_vec)
        return res

    def __str__(self) -> str:
        s = "<Robot[{}] name:{}, ndofs:{}>".format(
            self.__class__.__name__,
            self.name,
            self.n_dofs,
        )
        return s

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Forward Kinematics                                             ---
    # ---                                                                                                            ---

    def forward_kinematics(self, x: PT_NP_TYPE, solver="klampt") -> PT_NP_TYPE:
        """Interface for running forward kinematics.

        Args:
            x (PT_NP_TYPE): joint angles
            solver (str, optional): Solver to use. Options are "klampt" and "batchfk". The solver must be 'batchfk' if x
                                        is a torch.Tensor. Defaults to "klampt".
        """

        if isinstance(x, torch.Tensor):
            assert solver == "batchfk", "Only batchfk is supported for torch tensors"

        if solver == "klampt":
            return self.forward_kinematics_klampt(x)
        if solver == "batchfk":
            return self.forward_kinematics_batch(x, return_quaternion=True, return_runtime=False)
        raise ValueError(f"Solver '{solver}' not recognized")

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

        if self._joint_chain[0].name not in self._parent_T_joint_cache[device]:
            for joint in self._joint_chain:
                T = torch.diag_embed(torch.ones(1, 4, device=device, dtype=dtype))
                # TODO(@jstmn): Confirm that its faster to run `rpy_tuple_to_rotation_matrix` on the cpu and then send
                # to the gpu
                R = rpy_tuple_to_rotation_matrix(joint.origin_rpy, device=device)
                T[:, 0:3, 0:3] = R
                T[:, 0, 3] = joint.origin_xyz[0]
                T[:, 1, 3] = joint.origin_xyz[1]
                T[:, 2, 3] = joint.origin_xyz[2]
                self._parent_T_joint_cache[device][joint.name] = T.to(device)

    # TODO: Do FK starting at specific joint (like 'base_link') instead of the first joint.
    # TODO: Consider removing all cpu code from this function
    def forward_kinematics_batch(
        self,
        x: torch.Tensor,
        out_device: Optional[str] = None,
        dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
        return_quaternion: bool = True,
        return_runtime: bool = False,
        return_full_joint_fk: bool = False,
        return_full_link_fk: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Iterate through each joint in `self.joint_chain` and apply the joint's fixed transformation. If the joint is
        revolute then apply rotation x[i] and increment i

        Args:
            x (torch.tensor): [N x n_dofs] tensor, representing the joint angle values to calculate the robots FK with
            out_device (str): The device to save tensors to.
            return_quaternion (bool): Return format is [N x 7] where [:, 0:3] are xyz, and [:, 3:7] are quaternions.
                                        Otherwise return [N x 4 x 4] homogenious transformation matrices

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]:
                2. [N x 4 x 4] torch tensor of the transform between the parent link of the first joint and the end
                                effector of the robot
                3. The total runtime of the function. For convenience
        """
        assert self._batch_fk_enabled, f"BatchFK is disabled for '{self.name}'"
        # assert CUDA_AVAILABLE, f"forward_kinematics_batch() requires cuda"

        time0 = time()
        batch_size = x.shape[0]
        _assert_is_joint_angle_matrix(x, self.n_dofs)
        if out_device is None:
            out_device = x.device
        out_device_is_cpu = "cpu" in str(out_device)

        self._ensure_forward_kinematics_cache(out_device, dtype=dtype)

        # Rotation from body/base link to joint along the joint chain
        base_T_joint = torch.diag_embed(torch.ones(batch_size, 4, device=out_device, dtype=dtype))
        base_T_joints = [base_T_joint]
        # Almost the same as base_T_joints except for fixed joints
        base_T_links = [base_T_joint]

        # Iterate through each joint in the joint chain
        x_i = 0
        for joint in self._joint_chain:
            assert joint.joint_type not in UNHANDLED_JOINT_TYPES, f"Joint type '{joint.joint_type}' is not implemented"

            # translate + rotate joint frame by `origin_xyz`, `origin_rpy`
            fixed_rotation_dict = self._parent_T_joint_cache[out_device]
            parent_T_child_fixed = fixed_rotation_dict[joint.name].expand(batch_size, 4, 4)  # zero-copy expansion
            base_T_joint = base_T_joint.bmm(parent_T_child_fixed)

            if joint.joint_type in {"revolute", "continuous"}:
                # rotate the joint frame about the `axis_xyz` axis by `x[:, x_i]` radians
                rotation_amt = x[:, x_i]
                rotation_axis = joint.axis_xyz

                # TODO: Implement a more efficient approach than converting to rotation matrices. work just with rpy?
                # or quaternions?
                joint_rotation = single_axis_angle_to_rotation_matrix(
                    rotation_axis, rotation_amt, out_device=out_device
                )
                assert joint_rotation.shape == (batch_size, 3, 3)

                # TODO(@jstmn): determine which of these two implementations if faster
                T = torch.diag_embed(torch.ones(batch_size, 4, device=out_device, dtype=dtype))
                T[:, 0:3, 0:3] = joint_rotation
                base_T_joint = base_T_joint.bmm(T)
                base_T_joints.append(base_T_joint)
                base_T_links.append(base_T_joint)

                # Note: The line below can't be used because inplace operation is non differentiable. Keeping this here
                # as a reminder
                # base_T_joint[:, 0:3, 0:3] = base_T_joint[:, 0:3, 0:3].bmm(joint_rotation)

                x_i += 1

            elif joint.joint_type == "prismatic":
                # Note: [..., None] is a trick to expand the x[:, x_i] tensor.
                translations = (
                    torch.tensor(joint.axis_xyz, device=out_device, dtype=dtype) * x[:, x_i, None]
                )  # [batch x 3]
                assert translations.shape == (batch_size, 3)

                # TODO(@jstmn): consider making this more space efficient. create once and override?
                joint_fixed_T_joint = torch.diag_embed(torch.ones(batch_size, 4, device=out_device, dtype=dtype))
                joint_fixed_T_joint[:, 0:3, 3] = translations
                base_T_joint = base_T_joint.bmm(joint_fixed_T_joint)
                base_T_joints.append(base_T_joint)
                base_T_links.append(base_T_joint)

                x_i += 1

            elif joint.joint_type == "fixed":
                base_T_joints[-1] = base_T_joint
                base_T_links.append(base_T_joint)
            else:
                raise RuntimeError(f"Unhandled joint type {joint.joint_type}")

        if return_quaternion:
            quaternions = rotation_matrix_to_quaternion(base_T_joint[:, 0:3, 0:3])
            translations = base_T_joint[:, 0:3, 3]
            base_T_joint = torch.cat([translations, quaternions], dim=1)

        if return_runtime:
            return base_T_joint, time() - time0

        if return_full_joint_fk:
            ret = torch.stack(base_T_joints, dim=1)
            assert ret.shape == (
                batch_size,
                self.n_dofs + 1,
                4,
                4,
            ), f"ret.shape={ret.shape}"
            return ret

        # NOTe: DO NOT CHANGE THIS CODE!!! self-collision NN in cppflow depends on this format
        if return_full_link_fk:
            ret = torch.stack(base_T_links, dim=1)
            assert ret.shape == (
                batch_size,
                len(self._joint_chain) + 1,
                4,
                4,
            ), f"ret.shape={ret.shape} != ({batch_size}, {len(self._joint_chain)}, 4, 4)"
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
        _assert_is_joint_angle_matrix(xs, self.n_dofs)
        n = xs.shape[0]
        Js = np.zeros((n, 6, self.n_dofs))
        for i in range(n):
            Js[i] = self.jacobian_np(xs[i])
        return Js

    def jacobian_batch_pt(
        self,
        x: torch.Tensor,
        out_device: Optional[str] = None,
        dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
    ) -> torch.Tensor:
        """Return a batch of jacobian matrices for the given joint angle
        vectors. The rows are:

        0: roll (rx)
        1: pitch (ry)
        2: yaw (rz)
        3: x
        4: y
        5: z

        Args:
            x (np.ndarray): The joint angle to the jacobian with respect to

        Returns:
            np.ndarray: a [6 x ndof] matrix, where the top 3 rows are the orientation derivatives, and the bottom three
                        rows are the positional derivatives.
        """

        batch_size = x.shape[0]
        _assert_is_joint_angle_matrix(x, self.n_dofs)
        if out_device is None:
            out_device = x.device

        self._ensure_forward_kinematics_cache(out_device, dtype=dtype)

        # Rotation from body/base link to joint along the joint chain
        base_T_joints = self.forward_kinematics_batch(x, return_full_joint_fk=True, out_device=out_device, dtype=dtype)
        base_T_joints = base_T_joints[:, 1:, :, :]  # remove the base link

        # Compute jacobian
        J = torch.zeros((batch_size, 6, self.n_dofs), device=out_device, dtype=dtype)
        x_i = 0
        for joint in self._joint_chain:
            assert joint.joint_type not in UNHANDLED_JOINT_TYPES, f"Joint type '{joint.joint_type}' is not implemented"
            if joint.joint_type in {"revolute", "continuous"}:
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

    def inverse_kinematics_single_step_batch_np(
        self, target_poses: np.ndarray, xs_current: np.ndarray, alpha: float = 0.25
    ) -> Tuple[np.ndarray, float]:
        """Perform a single inverse kinematics step on a batch of joint angle vectors

        Note: the previous title for this function was jac_pinvstep_single_pose_np (see
                https://github.com/cbames/nn_ik/commit/d99fb10429f2529a94a284bfdafd08f8776877a4)

        Args:
            target_poses (np.ndarray): [batch x 7] poses to optimize the joint angles towards
            xs_current (np.ndarray): [batch x ndofs] joint angles to start the optimization from
            alpha (float, optional): Step size for the optimization step. Defaults to 0.25.

        Returns:
            Tuple[np.ndarray, float]: _description_
        """
        _assert_is_pose_matrix(target_poses)
        _assert_is_joint_angle_matrix(xs_current, self.n_dofs)
        assert xs_current.shape[0] == target_poses.shape[0]
        n = target_poses.shape[0]

        # Get the jacobian of the end effector with respect to the current joint angles
        J = self.jacobian_batch_np(xs_current)
        J_pinv = np.linalg.pinv(J)  # Jacobian pseudo-inverse

        # Run the xs_current through FK to get their realized poses
        current_poses = self.forward_kinematics_klampt(xs_current)
        assert current_poses.shape == target_poses.shape

        # Fill out `pose_errors` - the matrix of positional and rotational for each row (rotational error is in rpy)
        pose_errors = np.zeros((n, 6, 1))
        for i in range(3):
            pose_errors[:, i + 3, 0] = target_poses[:, i] - current_poses[:, i]

        current_pose_quat_inv = quaternion_inverse(current_poses[:, 3:7])
        rotation_error_quat = quaternion_product(target_poses[:, 3:], current_pose_quat_inv)
        rotation_error_rpy = quaternion_to_rpy(rotation_error_quat)  # check
        pose_errors[:, 0:3, 0] = rotation_error_rpy  #

        # tensor dimensions: [batch x ndofs x 6] * [batch x 6 x 1] = [batch x ndofs x 1]
        delta_x = J_pinv @ pose_errors
        xs_updated = xs_current + alpha * delta_x[:, :, 0]
        xs_updated = self.clamp_to_joint_limits(xs_updated)

        return xs_updated

    def inverse_kinematics_single_step_levenburg_marquardt(
        self,
        target_poses: torch.Tensor,
        xs_current: torch.Tensor,
        lambd: float = 0.0001,
    ) -> torch.Tensor:
        """Perform a levenburg-marquardt optimization step."""
        n = xs_current.shape[0]
        eye = torch.eye(self.n_dofs, device=xs_current.device)[None, :, :].repeat(n, 1, 1)

        # Get current error
        current_poses = self.forward_kinematics_batch(xs_current, out_device=xs_current.device, dtype=xs_current.dtype)
        # TODO: Use cat instead of creating a new tensor
        pose_errors = torch.zeros((n, 6, 1), device=xs_current.device, dtype=xs_current.dtype)  # [n 6 1]
        for i in range(3):
            pose_errors[:, i + 3, 0] = target_poses[:, i] - current_poses[:, i]
        current_pose_quat_inv = quaternion_inverse(current_poses[:, 3:7])
        rotation_error_quat = quaternion_product(target_poses[:, 3:], current_pose_quat_inv)
        rotation_error_rpy = quaternion_to_rpy(rotation_error_quat)
        pose_errors[:, 0:3, 0] = rotation_error_rpy  #

        J_batch = torch.tensor(
            self.jacobian_batch_np(np.array(xs_current.detach().cpu())),
            device=xs_current.device,
            dtype=xs_current.dtype,
        )  # [n 6 ndof]
        assert J_batch.shape == (n, 6, self.n_dofs)
        J_batch_T = torch.transpose(J_batch, 1, 2)  # [n ndof 6]
        assert J_batch_T.shape == (
            n,
            self.n_dofs,
            6,
        ), f"error, J_batch_T: {J_batch_T.shape}, should be {(n, self.n_dofs, 6)}"

        lfs_A = torch.bmm(J_batch_T, J_batch) + lambd * eye  # [n ndof ndof]
        rhs_B = torch.bmm(J_batch_T, pose_errors)  # [n ndof 1]
        delta_x = torch.linalg.solve(lfs_A, rhs_B)  # [n ndof 1]
        return xs_current + torch.squeeze(delta_x)

    # TODO: Enforce joint limits
    def inverse_kinematics_single_step_batch_pt(
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
        assert self._batch_fk_enabled, f"batch_fk is required for batch_ik, but is disabled for this robot"
        _assert_is_pose_matrix(target_poses)
        _assert_is_joint_angle_matrix(xs_current, self.n_dofs)
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
        current_poses = self.forward_kinematics_batch(xs_current, out_device=xs_current.device, dtype=dtype)
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
        assert self._batch_fk_enabled, f"batch_fk is required for batch_ik, but is disabled for this robot"
        _assert_is_pose_matrix(target_poses)
        _assert_is_joint_angle_matrix(xs_current, self.n_dofs)
        assert xs_current.shape[0] == target_poses.shape[0]
        assert (
            xs_current.device == target_poses.device
        ), f"xs_current and target_poses must be on the same device (got {xs_current.device} and {target_poses.device})"
        n = target_poses.shape[0]

        # New graph
        xs_current = xs_current.detach()
        xs_current.requires_grad = True

        # Run the xs_current through FK to get their realized poses
        current_poses = self.forward_kinematics_batch(xs_current, out_device=xs_current.device, dtype=dtype)
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
        assert pose.size == 7
        if seed is not None:
            _assert_is_np(seed, variable_name="seed")
            assert len(seed.shape) == 1, f"Seed must be a 1D array (currently: {seed.shape})"
            assert seed.size == self.n_dofs
            seed_q = self._x_to_qs(seed.reshape((1, self.n_dofs)))[0]

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

    def self_collision_distances_batch(self, x: torch.Tensor, use_qpth: bool = False) -> torch.Tensor:
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
        base_T_links = self.forward_kinematics_batch(x, return_full_link_fk=True, out_device=x.device, dtype=x.dtype)
        T1s = base_T_links[:, self._collision_idx0, :, :].reshape(-1, 4, 4)
        T2s = base_T_links[:, self._collision_idx1, :, :].reshape(-1, 4, 4)
        c1s = self._collision_capsules[self._collision_idx0, :].expand(batch_size, -1, -1).reshape(-1, 7)
        c2s = self._collision_capsules[self._collision_idx1, :].expand(batch_size, -1, -1).reshape(-1, 7)

        dists = capsule_capsule_distance_batch(c1s, T1s, c2s, T2s, use_qpth).reshape(batch_size, -1)

        return dists

    def self_collision_distances_jacobian_batch(self, x: torch.Tensor) -> torch.Tensor:
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
            dual_output = self.self_collision_distances_batch(dual_input)
            J = torch.autograd.forward_ad.unpack_dual(dual_output).tangent
            ndists = J.shape[1]
            J = J.reshape(nbatch, ndofs, ndists).permute(0, 2, 1)

            return J

    def env_collision_distances_batch(
        self, x: torch.Tensor, cuboid: torch.Tensor, Tcuboid: torch.Tensor
    ) -> torch.Tensor:
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
        base_T_links = self.forward_kinematics_batch(x, return_full_link_fk=True, out_device=x.device, dtype=x.dtype)
        Tcapsules = base_T_links.reshape(-1, 4, 4)
        big_batch_size = Tcapsules.shape[0]
        capsules = self._collision_capsules.expand(batch_size, -1, -1).reshape(-1, 7)
        Tcuboid = Tcuboid.expand(big_batch_size, 4, 4)
        cuboid = cuboid.expand(big_batch_size, 6)

        dists = capsule_cuboid_distance_batch(capsules, Tcapsules, cuboid, Tcuboid).reshape(batch_size, -1)

        return dists

    def env_collision_distances_jacobian_batch(
        self, x: torch.Tensor, cuboid: torch.Tensor, Tcuboid: torch.Tensor
    ) -> torch.Tensor:
        nbatch = x.shape[0]
        ndofs = x.shape[1]

        with torch.autograd.forward_ad.dual_level():
            dual_input = torch.autograd.forward_ad.make_dual(
                x.unsqueeze(1).expand(nbatch, ndofs, ndofs).reshape(nbatch * ndofs, ndofs).clone(),
                torch.eye(ndofs, device=x.device).expand(nbatch, ndofs, ndofs).reshape(-1, ndofs).clone(),
            )
            dual_output = self.env_collision_distances_batch(dual_input, cuboid, Tcuboid)
            J = torch.autograd.forward_ad.unpack_dual(dual_output).tangent
            ndists = J.shape[1]
            J = J.reshape(nbatch, ndofs, ndists).permute(0, 2, 1)

            return J


def forward_kinematics_kinpy(robot: Robot, x: np.array) -> np.array:
    """
    Returns the pose of the end effector for each joint parameter setting in x
    """
    _assert_is_joint_angle_matrix(x, robot.n_dofs)

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
        for i in range(robot.n_dofs):
            fk_dict[robot.actuated_joint_names[i]] = xs[i]
        return fk_dict

    for i in range(n):
        th = get_fk_dict(x[i])
        transform = kinpy_fk_chain.forward_kinematics(th, world=zero_transform)[robot.end_effector_link_name]
        y[i, 0:3] = transform.pos
        y[i, 3:] = transform.rot
    return y
