from jrl.robot import Robot







class WorldModel:

    def __init__(self, robot: Robot, obstacles: BatchedCuboidObstacles):
        self._robot = robot
        self._obstacles = obstacles