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

def pairwise_int(dataset, paired_batches, args):
    o_dir = os.path.join(args.o, args.i, "pairwise")
    for pair in paired_batches:
        batch1, batch2 = pair[0], pair[1]
        task_box((pair[0], pair[1]))
        data_i = dataset[dataset.obs[args.b].isin([batch1, batch2])]
        data_i = scib.preprocessing.hvg_batch(
            data_i,
            batch_key=args.b,
            target_genes=args.h,
            adataOut=True
        )
        file_path = f"{args.o}/{args.i}/pairwise/SCITUNA_[{batch1}]_[{batch2}].h5ad"
        if os.path.isfile(file_path):
            continue
        ifile_path = f"{args.o}/{args.i}/pairwise/IN_[{batch1}]_[{batch2}].h5ad"
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
            "--t", "pairwise"
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


def main(args):
    header()

    o_dir = os.path.join(args.o, args.i, "pairwise")
    os.makedirs(o_dir, exist_ok=True)

    #load dataset
    print("Load dataset...")
    try:
        original_dataset = scanpy.read_h5ad("data/{}_unintegrated.h5ad".format(args.i))
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
    pairwise_int(original_dataset, paired_batches, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SCITUNA")
    parser.add_argument("--i", required=True, help="input dataset")
    parser.add_argument("--b", required=True, help="batch")
    parser.add_argument("--h", required=False, default=2000, help="#HVGs")
    parser.add_argument("--o", required=False, default="output", help="output location")
    parser.add_argument("--c", required=False, help="number of cores (parallelism)")
    args = parser.parse_args()
    main(args)
