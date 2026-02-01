"""Script to download ImageNet class index mapping."""

import json
import urllib.request
import os

def download_imagenet_labels():
    """Download ImageNet class index JSON file.
    
    Downloads the ImageNet class index mapping and saves it to the data
    directory for use by the imaging utilities.
    
    Returns:
        Path to the downloaded file if successful, None otherwise.
    """
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    
    # Get the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'imagenet_class_index.json')
    
    print(f"Downloading ImageNet class index from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading: {e}")
        print("You can manually download the file from:")
        print(url)
        print(f"And save it to: {output_path}")
        return None

if __name__ == "__main__":
    download_imagenet_labels()
