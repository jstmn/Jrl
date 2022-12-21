from typing import Tuple
import unittest

from jkinpylib.robot import Robot
from jkinpylib.robots import PandaArm
from jkinpylib.math_utils import geodesic_distance_between_quaternions
from jkinpylib.utils import set_seed

import torch
import numpy as np

# Set seed to ensure reproducibility
set_seed()

np.set_printoptions(edgeitems=30, linewidth=100000)


class TestSolutionRerfinement(unittest.TestCase):
    @classmethod
    def setUpClass(clc):
        clc.panda_arm = PandaArm()

    def test_ik_with_random_seed(self):
        """Test that ik steps are making progress"""

        zero_x = np.zeros((1, 7))

        # pose_zero_x =
        # goal_config =


if __name__ == "__main__":
    unittest.main()
