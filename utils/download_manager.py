import requests
import os
from tqdm import tqdm
import git
from huggingface_hub import snapshot_download

def download_file(url, save_path):
    """
    Downloads a file from a URL with a progress bar, chunking, and resume support.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path to save the downloaded file.

    Returns:
        str: The path to the downloaded file, or None if the download fails.

    Raises:
        requests.exceptions.RequestException: If there is an error with the HTTP request.
        IOError: If there is an error writing to the file.
    """
    try:
        response = requests.head(url)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            if file_size < total_size_in_bytes:
                initial_pos = file_size
                headers = {'Range': f'bytes={initial_pos}-'}
                response = requests.get(url, stream=True, headers=headers)
                mode = 'ab'  # append mode
            else:
                print(f"File already fully downloaded: {save_path}")
                return save_path
        else:
            initial_pos = 0
            response = requests.get(url, stream=True)
            mode = 'wb'  # write mode

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        chunk_size = 1024
        with open(save_path, mode) as file, tqdm(
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
            initial=initial_pos,
            desc=save_path.split('/')[-1]
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
                    pbar.update(len(chunk))
        return save_path
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        return None
    except IOError as e:
        print(f"File I/O error: {e}")
        return None

def clone_repository(repo_url, save_path):
    """
    Clones a repository from a given URL (GitHub or Hugging Face).

    Args:
        repo_url (str): The URL of the repository to clone.  Supports GitHub and Hugging Face URLs.
        save_path (str): The local path to clone the repository to.

    Returns:
        str: The path to the cloned repository, or None if the cloning fails.

    Raises:
        ValueError: If the repository URL is not supported (not GitHub or Hugging Face).
        git.GitCommandError: If there is an error with the Git command.
        Exception: For any other unexpected errors.
    """
    try:
        if "github.com" in repo_url:
            class GitProgress(git.RemoteProgress):
                def __init__(self):
                    super().__init__()
                    self.pbar = None

                def update(self, op_code, cur_count, max_count=None, message=''):
                    if self.pbar is None:
                        self.pbar = tqdm(total=max_count, desc="Cloning repository")
                    self.pbar.total = max_count
                    self.pbar.desc = message
                    self.pbar.update(cur_count - self.pbar.n)

                def finalize(self):
                    if self.pbar:
                        self.pbar.close()

            progress = GitProgress()
            git.Repo.clone_from(repo_url, save_path, progress=progress)
            return save_path
        elif "huggingface.co" in repo_url or "huggingface.co/" in repo_url:
            repo_path = repo_url.replace("https://huggingface.co/", "")
            repo_id = repo_path
            repo_name = repo_id
            save_path = os.path.join(save_path, repo_name) if save_path else repo_name
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id, local_dir=save_path, repo_type="model", resume_download=True)
            return save_path
        else:
            raise ValueError("Unsupported repository URL. Only Hugging Face and GitHub are supported.")
    except git.GitCommandError as e:
        print(f"Repository cloning failed (Git): {e}")
        return None
    except ValueError as e:
        print(f"Invalid URL error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    pass
