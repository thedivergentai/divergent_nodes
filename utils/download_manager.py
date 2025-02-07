import requests
import os
from tqdm import tqdm

def download_file(url, destination, chunk_size=8192, resume_byte_pos=None):
    """Downloads a file with resuming and progress bar."""

    headers = {}
    if resume_byte_pos:
        headers['Range'] = f'bytes={resume_byte_pos}-'

    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes

    total_size = int(response.headers.get('content-length', 0))
    if resume_byte_pos:
        total_size += resume_byte_pos

    mode = 'ab' if resume_byte_pos else 'wb'
    with open(destination, mode) as file, tqdm(
        total=total_size,
        initial=resume_byte_pos if resume_byte_pos else 0,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading {os.path.basename(destination)}"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
                progress_bar.update(len(chunk))

def download_model(model_url, model_path):
    """Downloads a model, checking for existing files and resuming if necessary."""

    if os.path.exists(model_path):
        downloaded_size = os.path.getsize(model_path)
        download_file(model_url, model_path, resume_byte_pos=downloaded_size)
    else:
        download_file(model_url, model_path)
