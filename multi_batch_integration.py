import os
import gc
import scib
import scanpy
import warnings
import psutil
import anndata
import argparse
import threading
import subprocess
import numpy as np
import pandas as pd
from utils import *
import h5py
import sys
import resource


warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed

def multi_batch_int(dataset, args):

    current_adata = dataset.copy()
    current_adata = scib.preprocessing.hvg_batch(
        current_adata,
        batch_key=args.b,
        target_genes=args.h,
        adataOut=True
    )

    o_dir = os.path.join(args.o, args.i, "MBI")
    rts = {}
    otcs = {}
    pca = PCA(n_components=100, random_state=0)
    datasets_pca = {batch:
                        pca.fit_transform(
                            current_adata[current_adata.obs[args.b] == batch].X.toarray())
                    for batch in np.unique(current_adata.obs[args.b])}
    while len(np.unique(current_adata.obs[args.b])) > 1:
        (batch1, batch2), otcs = score_batches_parallal(current_adata, otcs, datasets_pca, args)
        task_box((batch1, batch2))
        file_path = f"{args.o}/{args.i}/MBI/SCITUNA_[{batch1}]_[{batch2}].h5ad"
        if os.path.isfile(file_path):
            data_i = scanpy.read_h5ad(file_path)

        else:

            data_i = current_adata[current_adata.obs[args.b].isin((batch1, batch2))]
            ifile_path = f"{args.o}/{args.i}/MBI/IN_[{batch1}]_[{batch2}].h5ad"

            scanpy.write(adata=data_i, filename=ifile_path)
            del data_i
            gc.collect()
            cmd = [
                "python", "run_scituna.py",
                "--i", args.i,
                "--f", ifile_path,
                "--b", args.b,
                "--o", args.o,
                "--c", str(args.c),
                "--t", "MBI"
            ]
            process = subprocess.Popen(cmd)
            ps_proc = psutil.Process(process.pid)


            process.wait()  # Blocks execution until the process is done

            # Ensure the process is properly terminated
            if process.poll() is None:  # Check if the process is still running
                process.terminate()  # Send a termination signal
                process.wait(timeout=10)  # Wait a bit for it to terminate

            # Ensure it's completely killed
            try:
                process.kill()  # Force kill if still running
            except Exception as e:
                print("Process already terminated:", e)

            # Optional: Free up memory if needed
            def free_up_memory():
                for proc in psutil.process_iter(attrs=['pid', 'name']):
                    if proc.info['pid'] == process.pid:
                        proc.terminate()
                        proc.wait()
                        print(f"Freed memory from process {proc.info['name']} (PID {process.pid})")

            free_up_memory()
            data_i = scanpy.read_h5ad(file_path)
        current_adata = current_adata[current_adata.obs[args.b] != batch1]
        current_adata = current_adata[current_adata.obs[args.b] != batch2]
        current_adata = anndata.concat([current_adata, data_i], join='outer')

        keys = list(otcs.keys())
        datasets_pca.pop(batch1)
        datasets_pca.pop(batch2)

        for i in keys:
            if i[0] in (batch1, batch2) or i[1] in (batch1, batch2):
                otcs.pop(i)
        try:
            datasets_pca[list(data_i.obs[args.b])[0]] = pca.fit_transform(data_i.X.A)
        except:
            datasets_pca[list(data_i.obs[args.b])[0]] = pca.fit_transform(data_i.X)
        del data_i
        gc.collect()

    if len(np.unique(current_adata.obs[args.b])) == 1:
        current_adata.obs[args.b] = dataset[current_adata.obs_names].obs[
            args.b]
        scanpy.write(adata=current_adata, filename=f"{args.o}/{args.i}/MBI/SCITUNA.h5ad")

def main(args):
    header()

    o_dir = os.path.join(args.o, args.i, "MBI")
    os.makedirs(o_dir, exist_ok=True)

    #load dataset
    print("Load dataset...")
    try:
        original_dataset = scanpy.read_h5ad("data/{}.h5ad".format(args.i))
    except:
        raise ValueError(
            f"Invalid Dataset."
        )
    if args.b not in original_dataset.obs:
        raise ValueError(
            f"Invalid Batch ID."
        )

    #retreive batch pairs as tuples
    paired_batches = [(a, b) for i, a in enumerate(np.unique(original_dataset.obs[args.b])) for
                   b in np.unique(original_dataset.obs[args.b])[i + 1:]]

    print("There are :", len(paired_batches), " batch pairs.")
    multi_batch_int(original_dataset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SCITUNA")
    parser.add_argument("--i", required=True, help="input dataset")
    parser.add_argument("--b", required=True, help="batch")
    parser.add_argument("--h", required=False, default=2000, help="#HVGs")
    parser.add_argument("--o", required=False, default="output", help="output location")
    parser.add_argument("--c", required=False, help="number of cores (parallelism)")
    args = parser.parse_args()
    main(args)
