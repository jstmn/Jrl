def kinpy_fk(
    kinpy_fk_chain: Chain, actuated_joints: List[str], end_effector_link_name: str, dim_x: int, x: np.array
) -> np.array:
    """
    Returns the pose of the end effector for each joint parameter setting in x
    """

    assert not isinstance(x, torch.Tensor), "torch tensors not supported"
    assert len(x.shape) == 2, f"x must be (m, n), currently: {x.shape}"

    n = x.shape[0]
    y = np.zeros((n, 7))
    zero_transform = kp.transform.Transform()
    fk_dict = {}
    for joint_name in actuated_joints:
        fk_dict[joint_name] = 0.0

    def get_fk_dict(xs):
        for i in range(dim_x):
            fk_dict[actuated_joints[i]] = xs[i]
        return fk_dict

    for i in range(n):
        th = get_fk_dict(x[i])
        transform = kinpy_fk_chain.forward_kinematics(th, world=zero_transform)[end_effector_link_name]
        t = transform.pos
        quat = transform.rot

        y[i, 0:3] = t
        y[i, 3:] = quat

    return y


def klampt_fk(
    klampt_robot: klampt.RobotModel,
    klampt_end_effector_link: klampt.RobotModelLink,
    robot_configs: List[List[float]],
) -> np.array:
    """
    Returns the pose of the end effector for each joint parameter setting in x. Forward kinemaitcs calculated with
    klampt
    """
    dim_y = 7
    n = len(robot_configs)
    y = np.zeros((n, dim_y))

    for i in range(n):
        q = robot_configs[i]
        klampt_robot.setConfig(q)

        R, t = klampt_end_effector_link.getTransform()
        y[i, 0:3] = np.array(t)
        y[i, 3:] = np.array(so3.quaternion(R))

    return y
