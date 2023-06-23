# Jrl

Jrl ('Jeremy's robotics library') is a robotics library containing robot models for popular robots as well as efficient, pytorch based *parallelized* implementations of forward kinematics, inverse kinematics, and robot-robot + robot-environment collision checking. Robot models include (run with `scripts/visualize_robot.py` to view):

1. Franka Panda
2. Iiwa 7dof manipulator
3. Fetch 8dof mobile manipulator

Available operations include (all part of the `Robot` class):

1. `forward_kinematics_batch()`: batched forward kinematics
1. `jacobian_batch_pt()`: batched manipulator jacobian calculation (change of the end effector's pose w.r.t. each joint angle) 
1. `inverse_kinematics_single_step_levenburg_marquardt()`: computes a single batched inverse kinematics step using Levenburg-Marquardt optimization
1. `inverse_kinematics_single_step_batch_pt()`: computes a single batched inverse kinematics step using the traditional jacobian pseudo-inverse method
1. `self_collision_distances_batch()`: batched self-collision distance calculation
2. `self_collision_distances_jacobian_batch()`: batched calculation of the jacobian of the pair-wise self-collision distances w.r.t. joint angles
3. `env_collision_distances_batch()`: batched self-environment distance calculation
4. `env_collision_distances_jacobian_batch()`: batched calculation of the jacobian of the self-environment distances w.r.t. joint angles

**Quickstart code.** This script will load a Panda robot model and then run forward and inverse kinematics on randomly sampled configs. See demo.py for the complete script, which includes robot-robot and robot-environment collision checking.

```python
from jrl.robots import Panda
from jrl.evaluation import pose_errors_cm_deg
import torch

def assert_poses_almost_equal(poses_1, poses_2):
    pos_errors_cm, rot_errors_deg = pose_errors_cm_deg(poses_1, poses_2)
    assert (pos_errors_cm.max().item() < 0.01) and (rot_errors_deg.max().item() < 0.1)

robot = Panda()
joint_angles, poses = robot.sample_joint_angles_and_poses(n=5, return_torch=True) # sample 5 random joint angles and matching poses

# Run forward-kinematics
poses_fk = robot.forward_kinematics_batch(joint_angles) 
assert_poses_almost_equal(poses, poses_fk)

# Run inverse-kinematics
ik_sols = joint_angles + 0.1 * torch.randn_like(joint_angles) 
for i in range(5):
    ik_sols = robot.inverse_kinematics_single_step_levenburg_marquardt(poses, ik_sols)
assert_poses_almost_equal(poses, robot.forward_kinematics_batch(ik_sols))
```


Note: This project uses the `w,x,y,z` format for quaternions.

## Installation

Recommended: clone the repo and install with pip
```
git clone https://github.com/jstmn/jrl.git && cd jrl/
pip install -e .
# or:
pip install -e ".[dev]"
```

Second option: Install from pypi (not recomended - the pypi version will likely be out of date until this project hardens)
``` bash
pip install jrl
```