import unittest
import torch
from src.models.cbma import CBMA

class TestCBMA(unittest.TestCase):
    def test_forward_shape(self):
        model = CBMA(input_dim=77)
        dummy_input = torch.randn(32, 100, 77)  # batch=32, seq_len=100
        output = model(dummy_input)
        self.assertEqual(output.shape, (32, 2))  # 二分类输出

if __name__ == '__main__':
    unittest.main()
