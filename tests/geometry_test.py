import unittest

import torch
import numpy as np

from jkinpylib.geometry import CapsuleGeometry, capsule_capsule_distance_batch

from jkinpylib.utils import set_seed, to_torch

# Set seed to ensure reproducibility
set_seed()


class TestGeometry(unittest.TestCase):
    def test_capsule_capsule_distance_batch(self):
        """Test capsule_capsule_distance_batch() returns the expected distance"""

        # Test 1: two height=2 capsules parallel to one another and pointing in +z. The distance between them should be
        # 1.5: their +z axis are 2 apart, each has a radius of .25, so the distance between their cylinder portions is
        # 1.5

        # Both are capsules with radius=0.25, height=2.0
        caps1 = torch.tensor([[0.25, 2.0]], device="cpu", dtype=torch.float32)
        caps2 = torch.tensor([[0.25, 2.0]], device="cpu", dtype=torch.float32)

        pose_c1 = torch.tensor([[-1, 0, 0, 1, 0, 0, 0]], device="cpu", dtype=torch.float32)
        pose_c2 = torch.tensor([[1, 0, 0, 1, 0, 0, 0]], device="cpu", dtype=torch.float32)

        returned = capsule_capsule_distance_batch(c1, pose_c1, c2, pose_c2)
        expected = 1.5


if __name__ == "__main__":
    unittest.main()
