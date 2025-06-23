import os
import gc
import scib
import time
import scanpy
import warnings
import psutil
import anndata
import argparse
import threading
import SCITUNA
import subprocess
import time
import os
import numpy as np
import pandas as pd
from utils import *
import h5py
import sys
import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed




warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed

def main(args):
    ctx = contextvars.copy_context()
    #load dataset
    print("Load dataset...")
    try:
        data_i = scanpy.read_h5ad(f"{args.f}")
    except:
        raise ValueError(
            f"Invalid Dataset."
        )

    batch1 = np.unique(data_i.obs[args.b])[0]
    batch2 = np.unique(data_i.obs[args.b])[1]

    scituna = SCITUNA.SCITUNA(data_i, args.b, o_dir=None)
    def init():
        scituna.reduce_dimensions(pca_dims=100)
        scituna.inter_intra_similarities()
        scituna.clustering(kc=30)
        scituna.construct_edges()

    def anchors():
        ctx.run(scituna.anchors_selection)

    with ThreadPoolExecutor(max_workers=int(args.c)) as executor:
        tasks = [
            executor.submit(init),
            executor.submit(anchors)
        ]

        for task in as_completed(tasks):
            task.result()
    executor.shutdown(wait=True)
    scituna.build_graphs(skip=5)
    scituna.integrate_datasets()

    data_i = data_i[scituna.cell_ids, scituna.gene_ids]
    data_i.X[:len(scituna.D_q)] = scituna.D_q
    file_path = f"{args.o}/{args.i}/{args.t}/SCITUNA_[{batch1}]_[{batch2}].h5ad"

    new_batch_id = f"{batch1}_{batch2}"
    data_i.obs[args.b] = new_batch_id

    scanpy.write(adata=data_i, filename=file_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SCITUNA")
    parser.add_argument("--i", required=True, help="input dataset")
    parser.add_argument("--f", required=True, help="input file")
    parser.add_argument("--o", required=False, default="output", help="output location")
    parser.add_argument("--c", required=False, help="number of cores (parallelism)")
    parser.add_argument("--b", required=True, help="batch")
    parser.add_argument("--t", required=True, help="MBI or pairwise")
    args = parser.parse_args()
    print(args)

    main(args)
