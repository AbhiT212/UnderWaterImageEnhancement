import unittest
import torch
from p2pnet.model import P2PNet

class TestModelForwardPass(unittest.TestCase):
    def test_forward_pass_shape(self):
        """Tests if the model's output shape matches the input shape."""
        batch_size = 4
        height, width = 256, 256
        base_channels = 32

        model = P2PNet(base_ch=base_channels)
        dummy_input = torch.randn(batch_size, 3, height, width)

        output = model(dummy_input)

        self.assertEqual(output.shape, dummy_input.shape)

    def test_output_range(self):
        """Tests if the model's output is in the [0, 1] range due to Sigmoid."""
        model = P2PNet(base_ch=16) # Use fewer channels to speed up test
        dummy_input = torch.randn(2, 3, 64, 64) # Use smaller size

        output = model(dummy_input)

        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_cpu_gpu_consistency(self):
        """Tests if model can run on both CPU and a GPU if available."""
        model = P2PNet(base_ch=16)
        dummy_input = torch.randn(1, 3, 64, 64)

        # CPU pass
        model.to('cpu')
        output_cpu = model(dummy_input.to('cpu'))
        self.assertEqual(output_cpu.device.type, 'cpu')

        # GPU pass if available
        if torch.cuda.is_available():
            model.to('cuda')
            output_gpu = model(dummy_input.to('cuda'))
            self.assertEqual(output_gpu.device.type, 'cuda')

if __name__ == '__main__':
    unittest.main()