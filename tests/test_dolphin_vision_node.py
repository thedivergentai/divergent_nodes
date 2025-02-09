import unittest
import torch
import sys
import os
#from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModelNotFoundError

# Add the parent directory to sys.path to allow importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dolphin_vision_node import DolphinVisionNode

class TestDolphinVisionNode(unittest.TestCase):

    def setUp(self):
        """Set up for the test cases."""
        self.node = DolphinVisionNode()
        result = self.node.load_model()
        self.assertIsNone(result)


    def test_generate_answer_invalid_image(self):
        """Test generate_answer with invalid image input."""
        result = self.node.generate_answer(image="not a tensor", prompt="test prompt")
        self.assertEqual(result, ("Invalid input: 'image' must be a PyTorch tensor.",))

    def test_generate_answer_invalid_prompt(self):
        """Test generate_answer with invalid prompt input."""
        result = self.node.generate_answer(image=torch.randn(1, 3, 224, 224), prompt=123)
        self.assertEqual(result, ("Invalid input: 'prompt' must be a string.",))

    def test_generate_answer_model_not_loaded(self):
        """Test generate_answer when the model is not loaded."""
        self.node.model = None
        self.node.tokenizer = None
        result = self.node.generate_answer(image=torch.randn(1, 3, 224, 224), prompt="test prompt")
        self.assertEqual(result, ("Model not loaded. Please check the node's configuration and ensure the model is loaded successfully.",))

    def test_generate_answer_success(self):
        """Test successful answer generation."""
        image = torch.randn(1, 3, 224, 224)  # Create a dummy image tensor
        prompt = "Describe this image."
        result = self.node.generate_answer(image=image, prompt=prompt)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], str)
        self.assertNotEqual(result[0], "")  # Check that the answer is not empty


if __name__ == '__main__':
    unittest.main()
