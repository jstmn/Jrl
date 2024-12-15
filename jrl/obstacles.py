


@dataclass
class CuboidObstacle:
    TODO


class BatchedCuboidObstacle:
    TODO



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
        TODO
        raise NotImplementedError("Implement self_collision_distances_jacobian")

    def env_collision_distances_jacobian(
        self, x: torch.Tensor, cuboid: torch.Tensor, Tcuboid: torch.Tensor
    ) -> torch.Tensor:
        TODO
        raise NotImplementedError("Implement self_collision_distances_jacobian")







