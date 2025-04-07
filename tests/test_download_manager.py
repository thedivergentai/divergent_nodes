import unittest
from unittest.mock import patch, mock_open, MagicMock
import sys
import os
import requests
import git

# Add the parent directory to sys.path to allow importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.download_manager import download_file, clone_repository

class TestDownloadManager(unittest.TestCase):

    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_download_file_success(self, mock_getsize, mock_exists, mock_file, mock_get):
        """Test successful file download."""
        mock_exists.return_value = False
        mock_get.return_value.status_code = 200
        mock_get.return_value.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value.headers = {'content-length': '1024'}

        result = download_file('http://example.com/file.txt', 'file.txt')
        self.assertEqual(result, 'file.txt')
        mock_file.assert_called_with('file.txt', 'wb')
        mock_file().write.assert_called()

    @patch('requests.get', side_effect=requests.exceptions.RequestException('Mocked error'))
    def test_download_file_failure(self, mock_get):
        """Test file download failure."""
        result = download_file('http://example.com/file.txt', 'file.txt')
        self.assertIsNone(result)

    @patch('git.Repo.clone_from')
    def test_clone_repository_github_success(self, mock_clone):
        """Test successful GitHub repository cloning."""
        mock_clone.return_value = MagicMock()
        result = clone_repository('https://github.com/user/repo.git', 'repo')
        self.assertEqual(result, 'repo')
        mock_clone.assert_called()

    @patch('utils.download_manager.snapshot_download')
    def test_clone_repository_huggingface_success(self, mock_snapshot_download):
        """Test successful Hugging Face repository cloning."""
        repo_id = 'cognitivecomputations/dolphin-vision-7b'
        expected_path = os.path.join('hf_repo', repo_id)
        mock_snapshot_download.side_effect = None
        with patch('os.makedirs') as mock_makedirs:
          result = clone_repository(f'https://huggingface.co/{repo_id}', 'hf_repo')
        self.assertEqual(result, expected_path)
        mock_snapshot_download.assert_called_with(repo_id=repo_id, local_dir=expected_path, repo_type="model", resume_download=True)

    def test_clone_repository_invalid_url(self):
        """Test cloning with an invalid URL."""
        result = clone_repository('http://invalid-url.com', 'repo')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
