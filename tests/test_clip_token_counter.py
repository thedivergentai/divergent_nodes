import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to allow importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_token_counter import CLIPTokenCounter

@patch('clip_token_counter.CLIPTokenizer.from_pretrained', side_effect=Exception('Mocked error'))
class TestCLIPTokenCounter(unittest.TestCase):

    def setUp(self):
        """Set up for the test cases."""
        pass

    def test_count_tokens_empty_string(self, mock_tokenizer):
        """Test token counting with an empty string."""
        counter = CLIPTokenCounter()
        result = counter.count_tokens(text="", tokenizer_name="openai/clip-vit-base-patch32")
        self.assertEqual(result, (0,))

    def test_count_tokens_normal_string(self, mock_tokenizer):
        """Test token counting with a normal string."""
        counter = CLIPTokenCounter()
        result = counter.count_tokens(text="This is a test string.", tokenizer_name="openai/clip-vit-base-patch32")
        self.assertGreater(result[0], 0)  # Exact count depends on the tokenizer

    def test_count_tokens_special_characters(self, mock_tokenizer):
        """Test token counting with special characters."""
        counter = CLIPTokenCounter()
        result = counter.count_tokens(text=r"!@#$%^&*()_+=-`~[]\{}|;':\",./<>?", tokenizer_name="openai/clip-vit-base-patch32")
        self.assertGreater(result[0], 0)

    def test_count_tokens_tokenizer_error(self, mock_tokenizer):
      """Test count_tokens when tokenizer loading fails"""
      counter = CLIPTokenCounter()
      result = counter.count_tokens(text="some text", tokenizer_name="some_tokenizer")
      self.assertEqual(result,(0,))

if __name__ == '__main__':
  unittest.main()
