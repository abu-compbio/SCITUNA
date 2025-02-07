import sys
import requests
import os

datasets =  {"pancreas":"https://figshare.com/ndownloader/files/24539828",
             "lung":"https://figshare.com/ndownloader/files/24539942",
             "small_atac_peaks":"https://figshare.com/ndownloader/files/25721792",
             "small_atac_windows":"https://figshare.com/ndownloader/files/25721795"}

def download_dataset(dataset_name):
    if dataset_name not in datasets:
        print("Dataset not found!")
        return

    url = datasets[dataset_name]
    filename = os.path.basename("{}_unintegrated.h5ad".format(dataset_name))
    
    # Download the dataset
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Dataset '{dataset_name}' downloaded successfully as '{filename}'")
    else:
        print(f"Failed to download dataset '{dataset_name}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    dataset_name = sys.argv[1]
    download_dataset(dataset_name)
