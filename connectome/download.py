import os
import json
import requests
from datetime import datetime
from tqdm import tqdm

# URLs
CONNECTIONS_URL = "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"
ROOT_IDS_URL = "https://zenodo.org/records/10676866/files/proofread_root_ids_783.npy?download=1"
ANNOTATIONS_URL = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
META_DIR = os.path.join(BASE_DIR, "data", "metadata")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File {os.path.basename(dest_path)} already exists. Skipping.")
        return
    
    print(f"Downloading to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = file.write(data)
            bar.update(size)

def main():
    print("--- Starting FlyWire Data Download ---")
    download_file(CONNECTIONS_URL, os.path.join(RAW_DIR, "proofread_connections_783.feather"))
    download_file(ROOT_IDS_URL, os.path.join(RAW_DIR, "proofread_root_ids_783.npy"))
    download_file(ANNOTATIONS_URL, os.path.join(META_DIR, "neuron_annotations.tsv"))
    
    manifest = {
        "version": "783",
        "download_date": datetime.now().isoformat(),
        "files": [
            "data/raw/proofread_connections_783.feather",
            "data/raw/proofread_root_ids_783.npy",
            "data/metadata/neuron_annotations.tsv"
        ]
    }
    
    with open(os.path.join(BASE_DIR, "data", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    print("\nDownload complete and manifest saved.")

if __name__ == "__main__":
    main()

