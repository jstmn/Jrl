from typing import List, Tuple, Dict, Optional
from time import time
from dataclasses import dataclass
from time import time
from more_itertools import locate

from jkinpylib.math_utils import R_from_rpy_batch, R_from_axis_angle, quaternion_to_rpy_batch
from jkinpylib import config

# from jkinpylib.math_utils import matrix_to_quaternion, quaternion_invert, quaternion_multiply # TODO: Find these functions
from jkinpylib.urdf_utils import get_joint_chain, UNHANDLED_JOINT_TYPES, Joint

import torch
import numpy as np
import kinpy as kp
import klampt
from klampt import IKSolver
from klampt.model import ik
from klampt.math import so3


@dataclass
class IKResult:
    n_steps: int
    runtime: float
    target_pose: torch.Tensor
    solutions: torch.Tensor


class Robot:
    def __init__(
        self,
        name: str,
        urdf_filepath: str,
        joint_path: List[str],
        end_effector_link_name: str,
        batch_fk_enabled: bool = True,
        verbose: bool = False,
    ):
        """_summary_

        Args:
            urdf_filepath (str): _description_
            joint_path (List[str]): The name of the joints that form the kinematic chain that is being represented. There
                                are no restrictions on where it starts. It must end at the end effector however. NOTE:
                                This may include fixed joints
            end_effector_link_name (str): _description_

        Raises:
            ValueError: _description_
        """
        self._name = name
        self._urdf_filepath = urdf_filepath
        self._end_effector_link_name = end_effector_link_name

        # Note: `_joint_chain`, `_actuated_joint_limits`, `_actuated_joint_names` only includes the joints that were
        # specified by the subclass. It does not include all actuated joints in the urdf
        self._joint_chain = get_joint_chain(self._urdf_filepath, joint_path, self._end_effector_link_name)
        self._actuated_joint_limits = [joint.limits for joint in self._joint_chain if joint.is_actuated]
        self._actuated_joint_names = [joint.name for joint in self._joint_chain if joint.is_actuated]
        self._batch_fk_enabled = batch_fk_enabled

        # Create and fill cache of fixed rotations between links.
        self._fixed_rotations = {}
        self.forward_kinematics_batch(
            torch.tensor(self.sample_joint_angles(500), device=config.device, dtype=torch.float32)
        )

        # Initialize klampt
        # Note: Need to save `_klampt_world_model` as a member variable otherwise you'll be doomed to get a segfault
        self._klampt_world_model = klampt.WorldModel()
        self._klampt_world_model.loadRobot(self._urdf_filepath)
        self._klampt_robot: klampt.robotsim.RobotModel = self._klampt_world_model.robot(0)
        self._klampt_ee_link: klampt.robotsim.RobotModelLink = self._klampt_robot.link(self._end_effector_link_name)
        self._klampt_config_dim = len(self._klampt_robot.getConfig())
        self._klampt_driver_vec_dim = self._klampt_robot.numDrivers()
        self._klampt_active_dofs = self._get_klampt_active_dofs()
        self._klampt_active_driver_idxs = self._get_klampt_active_driver_idxs()

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
    def end_effector_link_name(self) -> str:
        """Returns the name of the end effector link in the urdf"""
        return self._end_effector_link_name

    @property
    def n_dofs(self) -> int:
        return sum([1 for joint in self._joint_chain if joint.is_actuated])

    @property
    def actuated_joint_names(self) -> List[str]:
        return self._actuated_joint_names

    @property
    def actuated_joints_limits(self) -> List[Tuple[float, float]]:
        return self._actuated_joint_limits

    @property
    def klampt_world_model(self) -> klampt.WorldModel:
        return self._klampt_world_model

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             External Functions                                             ---
    # ---                                                                                                            ---

    def sample_joint_angles(self, n: int, solver=None) -> np.ndarray:
        """Returns a [N x ndof] matrix of randomly drawn joint angle vectors

        Args:
            n (int): _description_
            solver (_type_, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        angs = np.random.rand(n, self.n_dofs)  # between [0, 1)

        # Sample
        for i in range(self.n_dofs):
            range_ = self._actuated_joint_limits[i][1] - self._actuated_joint_limits[i][0]
            assert range_ > 0
            angs[:, i] *= range_
            angs[:, i] += self._actuated_joint_limits[i][0]
        return angs

    def set_klampt_robot_config(self, x: np.ndarray):
        """Set the internal klampt robots config with the given joint angle vector"""
        assert isinstance(x, np.ndarray), f"Expected x to be a numpy array but got {type(x)}"
        assert x.shape == (self.n_dofs,), f"Expected x to be of shape ({self.n_dofs},) but got {x.shape}"
        q = self._x_to_qs(np.resize(x, (1, self.n_dofs)))[0]
        self._klampt_robot.setConfig(q)

    def __str__(self) -> str:
        s = "<Robot[{}] name:{}, ndofs:{}>".format(
            self.__class__.__name__,
            self.name,
            self.n_dofs,
        )
        return s

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Internal Functions                                             ---
    # ---                                                                                                            ---

    def _get_klampt_active_dofs(self) -> List[int]:
        """Hack: We need to know which indexes of the klampt q vector are from active joints.

        Returns:
            List[int]: The indexes of the klampt configuration vector which correspond to the user specified active
                        joints
        """
        all_drivers = [self._klampt_robot.driver(i) for i in range(self._klampt_robot.numDrivers())]
        actuated_joint_child_names = [joint.child for joint in self._joint_chain]
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
        actuated_joint_child_names = [joint.child for joint in self._joint_chain]

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
            List[float]: A list with the joint angle vector formatted in the klampt driver format.
        """

        assert x.size == self.n_dofs, f"x doesn't have {self.n_dofs} (n_dofs) elements ({self.n_dofs} != {x.size})"
        assert x.shape == (self.n_dofs,), f"x.shape must be (n_dofs,) - ({(self.n_dofs,)}) != {x.shape}"

        # return x as a list if there are no additional active joints in the urdf
        if x.size == self._klampt_driver_vec_dim:
            return x.tolist()

        # TODO(@jstm): Consider a non iterative implementation for this
        driver_vec = [0] * self._klampt_driver_vec_dim

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
        assert isinstance(x, np.ndarray), f"Expected x to be a numpy array but got {type(x)}"
        assert len(x.shape) == 2, f"Expected x to be a 2D array but got {x.shape}"
        assert x.shape[1] == self.n_dofs, f"Expected x matrix width to be ndof {x.shape[1]} != {self.n_dofs}"

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

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Forward Kinematics                                             ---
    # ---                                                                                                            ---

    def forward_kinematics(self, x: np.array, solver="klampt") -> np.array:
        if solver == "klampt":
            return self.forward_kinematics_klampt(x)
        elif solver == "batchfk":
            return self.forward_kinematics_batch(x)
        else:
            raise ValueError(f"Solver '{solver}' not recognized")

    def forward_kinematics_klampt(self, x: np.array) -> np.array:
        """Forward kinematics using the klampt library"""
        robot_configs = self._x_to_qs(x)
        dim_y = 7
        n = len(robot_configs)
        y = np.zeros((n, dim_y))

        for i in range(n):
            q = robot_configs[i]
            self._klampt_robot.setConfig(q)
            R, t = self._klampt_ee_link.getTransform()
            y[i, 0:3] = np.array(t)
            y[i, 3:] = np.array(so3.quaternion(R))
        return y

    def forward_kinematics_batch(
        self, x: torch.tensor, device: str = config.device, dtype=torch.float32
    ) -> Tuple[torch.Tensor, float]:
        """Iterate through each joint in `self.joint_chain` and apply the joint's fixed transformation. If the joint is
        revolute then apply rotation x[i] and increment i

        Args:
            x (torch.tensor): [N x n_dofs] tensor, representing the joint angle values to calculate the robots FK with
            device (str): The device to save tensors to.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]:
                2. [N x 4 x 4] torch tensor of the transform between the parent link of the first joint and the end
                                effector of the robot
                3. The total runtime of the function. For convenience
        """
        if not self._batch_fk_enabled:
            raise NotImplementedError()

        time0 = time()
        batch_size = x.shape[0]
        assert x.shape[1] == self.n_dofs, f"Expected x matrix width to be ndof {x.shape[1]} != {self.n_dofs}"
        assert str(x.device) == str(device), f"Expected x to be on device '{device}' but got '{x.device}'"

        # Update _fixed_rotations if this is a larger batch then we've seen before
        if (
            self._joint_chain[0].name not in self._fixed_rotations
            or self._fixed_rotations[self._joint_chain[0].name].shape[0] < batch_size
        ):
            for joint in self._joint_chain:
                T = torch.diag_embed(torch.ones(batch_size, 4, device=device, dtype=dtype))
                # TODO(@jstmn): Confirm that its faster to run `R_from_rpy_batch` on the cpu and then send to the gpu
                R = R_from_rpy_batch(joint.origin_rpy, device="cpu")
                T[:, 0:3, 0:3] = R.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
                T[:, 0, 3] = joint.origin_xyz[0]
                T[:, 1, 3] = joint.origin_xyz[1]
                T[:, 2, 3] = joint.origin_xyz[2]
                self._fixed_rotations[joint.name] = T

        # Rotation from body/base link to joint along the joint chain
        base_T_joint = torch.diag_embed(torch.ones(batch_size, 4, device=device, dtype=dtype))

        # Iterate through each joint in the joint chain
        x_i = 0
        for joint in self._joint_chain:
            assert joint.joint_type not in UNHANDLED_JOINT_TYPES, f"Joint type '{joint.joint_type}' is not implemented"

            # translate + rotate joint frame by `origin_xyz`, `origin_rpy`
            parent_T_child_fixed = self._fixed_rotations[joint.name][0:batch_size]
            base_T_joint = base_T_joint.bmm(parent_T_child_fixed)
            assert base_T_joint.shape == (batch_size, 4, 4)

            if joint.joint_type == "revolute" or joint.joint_type == "continuous":
                # rotate the joint frame about the `axis_xyz` axis by `x[:, x_i]` radians
                rotation_amt = x[:, x_i]
                rotation_axis = joint.axis_xyz
                joint_rotation = R_from_axis_angle(
                    rotation_axis, rotation_amt, device=device
                )  # TODO: Implement a more efficient approach than converting to rotation matrices. work just with rpy? or quaternions?
                assert joint_rotation.shape == (batch_size, 3, 3)

                # TODO(@jstmn): determine which of these two implementations if faster
                # T = torch.diag_embed(torch.ones(batch_size, 4, device=device, dtype=dtype))
                # T[:, 0:3, 0:3] = joint_rotation
                # base_T_joint = base_T_joint.bmm(T)

                base_T_joint[:, 0:3, 0:3] = base_T_joint[:, 0:3, 0:3].bmm(joint_rotation)

            elif joint.joint_type == "prismatic":
                # Note: [..., None] is a trick to expand the x[:,x_i] tensor.
                translations = torch.tensor(joint.axis_xyz, device=device, dtype=dtype) * x[:, x_i, None]  # [batch x 3]
                assert translations.shape == (batch_size, 3)

                # TODO(@jstmn): consider making this more space efficient
                joint_fixed_T_joint = torch.diag_embed(torch.ones(batch_size, 4, device=device, dtype=dtype))
                joint_fixed_T_joint[:, 0:3, 3] = translations
                base_T_joint = base_T_joint.bmm(joint_fixed_T_joint)

            elif joint.joint_type == "fixed":
                pass
            else:
                raise RuntimeError(f"Unhandled joint type {joint.joint_type}")

            x_i += 1

        assert base_T_joint.shape == (batch_size, 4, 4)
        return base_T_joint, time() - time0

    # ------------------------------------------------------------------------------------------------------------------
    # ---                                                                                                            ---
    # ---                                             Inverse Kinematics                                             ---
    # ---                                                                                                            ---

    def jacobian(self, x: torch.tensor) -> np.ndarray:
        raise NotImplementedError()

    # see https://github.com/cbames/nn_ik/commit/d99fb10429f2529a94a284bfdafd08f8776877a4
    def inverse_kinematics_single_step_batch(
        self,
        target_pose: torch.Tensor,
        n_steps: int,
        alpha: float,
        device: str,
        x_init: Optional[torch.Tensor] = None,
    ) -> IKResult:
        """_summary_

        original title: jac_pinvstep_single_pose_np

        Args:
            robot (Robot): _description_
            target_pose (torch.Tensor): _description_
            n_steps (int): _description_
            alpha (float): _description_
            device (str): _description_
            x_init (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            IKResult: _description_
        """
        N = target_pose.shape[0]
        dofs = self.n_dofs
        target_pose_np = target_pose.cpu().detach().numpy()

        x = x_init
        if x is None:
            x = torch.rand((N, dofs), device=device)

        # for step_i in range(n_steps):

        # Compute the forward kinematics
        x_fk_t, x_fk_R, _ = self.forward_kinematics_batch(x, device="cpu")
        J = self.jacobian(x[:, 0:dofs])

        # Jacobian pseudo-inverse
        J_pinv = np.linalg.pinv(J)

        #
        y_deltas = np.zeros((N, 6))
        for i in range(3):
            y_deltas[:, i] = target_pose_np[i] - x_fk_t[:, i]

        y_target_quat = torch.from_numpy(np.tile(target_pose_np[3:], (N, 1)))
        x_fk_quat = matrix_to_quaternion(x_fk_R)
        x_fk_quat_inv = quaternion_invert(x_fk_quat)
        delta_quat = quaternion_multiply(y_target_quat, x_fk_quat_inv)
        delta_rpy = quaternion_to_rpy_batch(delta_quat)
        y_deltas[:, 3:] = delta_rpy

        delta_x = np.reshape(J_pinv @ np.reshape(y_deltas, (N, 6, 1)), (N, dofs))
        x = x[:, 0:dofs] + alpha * delta_x
        return x

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
            assert isinstance(seed, np.ndarray), f"seed must be a numpy array (currently {type(seed)})"
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
            # TODO(@jstmn): What does 'tolarance' mean for klampt? Positional error? Positional error + orientation error?
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
                        pose, seed=None, positional_tolerance=positional_tolerance, verbosity=verbosity
                    )

                continue

            if verbosity > 1:
                print("Solved in", solver.lastSolveIters(), "iterations")
                residual = solver.getResidual()
                print("Residual:", residual, " - L2 error:", np.linalg.norm(residual[0:3]))

            return self._qs_to_x([self._klampt_robot.getConfig()])

        if verbosity > 0:
            print("inverse_kinematics_klampt() - Failed to find IK solution after", n_tries, "optimization attempts")
        return None


# TODO: Where to move this? The klampt FK function is much faster
def forward_kinematics_kinpy(robot, x: np.array) -> np.array:
    """
    Returns the pose of the end effector for each joint parameter setting in x
    """
    assert len(x.shape) == 2, f"x must be (m, n), currently: {x.shape}"

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


"""
def jac_pinvstep_single_pose_np(model_wrapper, x: np.array, y_targets_np, alpha) -> config.MakeSamplesResult:
	Perform
	#TODO(@jeremysm): Working for single value, failing for batch of values

	N = x.shape[0] if len(x.shape) > 1 else 1
	x_torch = torch.FloatTensor(x, device="cpu")

	# Compute the forward kinematics
	x_fk_t, x_fk_R = model_wrapper.batch_fk_calc.batch_fk(x_torch, device="cpu")

	# Jacobian
	# TODO(@jeremysm): jacobian must be wrong
	J = model_wrapper.robot_model.jacobian(x[:, 0:model_wrapper.dim_x])

	# Jacobian pseudo-inverse
	J_pinv = np.linalg.pinv(J)

	#
	y_deltas = np.zeros((N, 6))
	for i in range(3):
		y_deltas[:, i] = y_targets_np[i] - x_fk_t[:, i]

	y_target_quat = torch.from_numpy(np.tile(y_targets_np[3:], (N, 1)))
	x_fk_quat = pytorch3d.transforms.matrix_to_quaternion(x_fk_R)
	x_fk_quat_inv = pytorch3d.transforms.quaternion_invert(x_fk_quat)
	delta_quat = pytorch3d.transforms.quaternion_multiply(y_target_quat, x_fk_quat_inv)
	delta_rpy = quaternion_to_rpy_batch(delta_quat)
	y_deltas[:, 3:] = delta_rpy

	delta_x = np.reshape(J_pinv @ np.reshape(y_deltas, (N, 6, 1)), (N, model_wrapper.dim_x))
	x = x[:, 0:model_wrapper.dim_x] + alpha * delta_x
	return ret
"""