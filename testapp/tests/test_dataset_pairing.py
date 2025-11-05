import unittest
import os
import shutil
import torch
from p2pnet.dataset import UnderwaterDataset

class TestDatasetPairing(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory with dummy image files."""
        self.test_dir = "temp_test_data"
        os.makedirs(os.path.join(self.test_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "gt"), exist_ok=True)

        # Create dummy files
        self.files = ["a.png", "b.png", "c.png"]
        for fname in self.files:
            open(os.path.join(self.test_dir, "raw", fname), 'a').close()
            open(os.path.join(self.test_dir, "gt", fname), 'a').close()

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_perfect_pairing(self):
        """Test if dataset pairs correctly when basenames match."""
        dataset = UnderwaterDataset(root_dir=self.test_dir)
        self.assertEqual(len(dataset), 3)
        for i, (raw_path, gt_path) in enumerate(dataset.image_pairs):
            self.assertEqual(os.path.basename(raw_path), self.files[i])
            self.assertEqual(os.path.basename(gt_path), self.files[i])

    def test_mismatched_pairing_fallback(self):
        """Test fallback to sorted lists when basenames don't match."""
        # Create a mismatched file
        os.rename(
            os.path.join(self.test_dir, "gt", "c.png"),
            os.path.join(self.test_dir, "gt", "d.png")
        )
        dataset = UnderwaterDataset(root_dir=self.test_dir)
        self.assertEqual(len(dataset), 3)
        # It should fall back to sorted order pairing
        raw_names = sorted([os.path.basename(p) for p, g in dataset.image_pairs])
        gt_names = sorted([os.path.basename(g) for p, g in dataset.image_pairs])
        self.assertEqual(raw_names, ["a.png", "b.png", "c.png"])
        self.assertEqual(gt_names, ["a.png", "b.png", "d.png"])

    def test_uneven_files_fallback(self):
        """Test fallback when number of files is different."""
        # Remove a gt file
        os.remove(os.path.join(self.test_dir, "gt", "c.png"))
        dataset = UnderwaterDataset(root_dir=self.test_dir)
        self.assertEqual(len(dataset), 2) # Should be trimmed to the minimum length

if __name__ == '__main__':
    unittest.main()