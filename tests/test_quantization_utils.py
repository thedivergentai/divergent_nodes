import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import torch

# Add the parent directory to sys.path to allow importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quantization_utils import quantize_model, unload_model, _apply_bitsandbytes_quantization


class TestQuantizationUtils(unittest.TestCase):

    def setUp(self):
        pass

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('bitsandbytes.nn.Linear8bitLt')
    @patch('bitsandbytes.nn.Linear4bit')
    def test_quantize_model_success(self, mock_linear4bit, mock_linear8bit, mock_from_pretrained):
        """Test successful model quantization."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        mock_linear8bit.return_value = mock_model
        mock_linear4bit.return_value = mock_model

        model = quantize_model('model_name', quantization_method='bitsandbytes', device='cpu', bits=8)
        self.assertIsNotNone(model)
        mock_from_pretrained.assert_called()
        mock_linear8bit.assert_called()

        model = quantize_model('model_name', quantization_method='bitsandbytes', device='cpu', bits=4)
        self.assertIsNotNone(model)
        mock_from_pretrained.assert_called()
        mock_linear4bit.assert_called()


    @patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=Exception('Mocked error'))
    def test_quantize_model_failure(self, mock_from_pretrained):
        """Test model quantization failure."""
        model = quantize_model('model_name', quantization_method='bitsandbytes', device='cpu', bits=8)
        self.assertIsNone(model)

    @patch('bitsandbytes.nn.Linear8bitLt')
    @patch('bitsandbytes.nn.Linear4bit')
    def test_apply_bitsandbytes_quantization_invalid_bits(self, mock_linear4bit, mock_linear8bit):
        """Test _apply_bitsandbytes_quantization with invalid bits value."""
        model = MagicMock()
        result = _apply_bitsandbytes_quantization(model, 'cpu', bits=16)
        self.assertIsNone(result)

    def test_unload_model(self):
        """Test model unloading."""
        model = MagicMock()
        unload_model(model)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
