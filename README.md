# Jrl

Jrl ('Jeremy's robotics library') is a robotics library containing robot models for popular robots as well as efficient, pytorch based *parallelized* implementations of forward kinematics, inverse kinematics, and robot-robot + robot-environment collision checking. 


**Robots**

Robot models include (run with `scripts/visualize_robot.py` to view):

| jrl name | full name                   |
|----------|-----------------------------|
| Panda    | Franka Panda                |
| Fetch    | Fetch                       |
| FetchArm | Fetch - Arm (no lift joint) |
| Iiwa7    | Kuka LBR IIWA7              |
| Rizon4   | Flexiv Rizon 4              |
| Ur5      | Ur5                         |


**Functions**

Available operations include (all part of the `Robot` class):

| function                           | description                                                                  |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `forward_kinematics_batch()`                           | (batched) forward kinematics                                                                  |
| `jacobian_batch_pt()`                                  | (batched) Jacobian of the manipulators forward kinematics map (w.r.t. joint angles)           |
| `inverse_kinematics_single_step_levenburg_marquardt()` | (batched) Inverse kinematics step using Levenburg-Marquardt                                   |
| `inverse_kinematics_single_step_batch_pt()`            | (batched) Inverse kinematics step using the jacobian pseudo-inverse method                    |
| `self_collision_distances_batch()`                     | (batched) Pairwise distance between each link of the robot                                    |
| `self_collision_distances_jacobian_batch()`            | (batched) Jacobian of `self_collision_distances_batch()` w.r.t. joint angles                  |
| `env_collision_distances_batch()`                      | (batched) Pairwise distance between each link of the robot and each cuboid in the environment |
| `env_collision_distances_jacobian_batch()`             | (batched) Jacobian of `env_collision_distances_batch()` w.r.t. joint angles                   |






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

Clone the repo and install with poetry. Don't use the version on pypi - it will remain out of date until this project hardens
```
git clone https://github.com/jstmn/jrl.git && cd jrl/
poetry install --without dev
# or:
poetry install # includes dev dependencies, like the linter
```