import os
import requests
from tqdm import tqdm
from torchvision import datasets

def download_file(url, local_filename):
    """
    Downloads a file from a given URL if it does not already exist and displays a progress bar.

    Args:
    url (str): URL of the file to download.
    local_filename (str): Local path where the file will be saved.
    """
    if not os.path.exists(local_filename):
        print(f"Downloading {url} to {local_filename}...")
        with requests.get(url, verify=False, stream=True) as response:
            response.raise_for_status()  # Check for bad status codes

            # Get the total file size from the headers
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            
            # Initialize the progress bar
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))  # Update progress bar with the size of the chunk
                    f.write(chunk)
            
            # Ensure the progress bar is closed upon completion
            progress_bar.close()
            
            # Check if the total size is known and if we have downloaded the entire file.
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")
        
        print(f"Downloaded {local_filename}")
    else:
        print(f"{local_filename} already exists. Skipping download.")

# URLs of the files to download
urls = [
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
]

# Local filenames to save the downloaded files
local_filenames = [
    "data/imagenet/train/ILSVRC2012_img_train.tar",
    "data/imagenet/val/ILSVRC2012_img_val.tar"
]

for url, local_filename in zip(urls, local_filenames):
    download_file(url, local_filename)


# train_dataset = datasets.ImageFolder('imagenet/train', transform=None)
# val_dataset = datasets.ImageFolder('imagenet/val', transform=None)