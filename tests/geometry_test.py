import unittest

import torch
import numpy as np

from jkinpylib.geometry import CapsuleGeometry, capsule_capsule_distance

from jkinpylib.utils import set_seed, to_torch

# Set seed to ensure reproducibility
set_seed()

# suppress=True: print with decimal notation, not scientific
# np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, precision=12)
# torch.set_printoptions(linewidth=5000, precision=8, sci_mode=False)


class TestGeometry(unittest.TestCase):
    def test_capsule_capsule_distance(self):
        """Test capsule_capsule_distance() returns the expected distance"""

        # Test 1: two height=2 capsules parallel to one another and pointing in +z

        c1 = CapsuleGeometry(radius=0.25, height=2.0)
        c2 = CapsuleGeometry(radius=0.25, height=2.0)

        pose_c1 = torch.tensor([-1, 0, 0, 1, 0, 0, 0], device="cpu", dtype=torch.float32)[None, :]
        pose_c2 = torch.tensor([1, 0, 0, 1, 0, 0, 0], device="cpu", dtype=torch.float32)[None, :]

        returned = capsule_capsule_distance(c1, pose_c1, c2, pose_c2)
        expected = 1.5


if __name__ == "__main__":
    unittest.main()
