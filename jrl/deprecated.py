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
