import os
import umap
import scanpy
import pickle
import warnings
import operator
import itertools
import umap.plot
import numpy as np
import pandas as pd
from utils import *
import seaborn as sns
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.gridspec import GridSpec
from sklearn.metrics.pairwise import cosine_similarity
import pyfiglet
from ot import emd2
from scipy.spatial.distance import cdist
from multiprocessing import Pool, Manager
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

def OTC(data1, data2):
    Ui = np.ones(data1.shape[0]) / data1.shape[0]
    Uj = np.ones(data2.shape[0]) / data2.shape[0]
    CMij = cdist(data1, data2, metric='euclidean')
    otc = emd2(Ui, Uj, CMij)
    return otc

def process_pair(pair, datasets_pca, otcs):
    if pair in otcs:
        return
    otcs[pair] = OTC(datasets_pca[pair[0]], datasets_pca[pair[1]])

def parallel_otcs(paired_batches, datasets_pca):
    with Manager() as manager:
        otcs = manager.dict()  # Shared dictionary
        with Pool() as pool:
            list(tqdm(pool.starmap(process_pair, [(pair, datasets_pca, otcs) for pair in paired_batches]),
                      total=len(paired_batches)))
        return dict(otcs)  # Convert back to a regular dictionary

def score_batches_parallal(anndata, otcs, datasets_pca, args):
    paired_batches = [(a, b) for i, a in enumerate(np.unique(anndata.obs[args.b])) for
                      b in np.unique(anndata.obs[args.b])[i + 1:]]
    otcs = parallel_otcs(paired_batches, datasets_pca)
    sorted(otcs.items(), key=lambda x: x[1], reverse=False)
    return sorted(otcs.items(), key=lambda x: x[1], reverse=False)[0][0], otcs

def header():
    print(pyfiglet.figlet_format("SCITUNA v1.0"))

def task_box(pair):
    s_style = f"Integrating: {pair[0]} and {pair[1]}"
    box_width = len(s_style) + 6  # Add extra space for padding
    top_bottom_border = "╔" + "═" * (box_width - 2) + "╗"  # Top border
    middle_border = "║" + " " * (box_width - 2) + "║"  # Side borders
    bottom_border = "╚" + "═" * (box_width - 2) + "╝"  # Bottom border

    print(top_bottom_border)
    print(middle_border)
    print(f"║  {s_style}  ║")  # Center the text with padding
    print(middle_border)
    print(bottom_border)

def load_and_combine_scores(outputs_folder, methods, datasets):
    methods_vs_metrics = {}
    for dataset in tqdm(datasets):
        for method in methods:
            try:
                method_scores = pd.read_csv("{}/{}/metrics/{}.csv".format(outputs_folder,dataset,method),index_col=0)
            except:
                print("Error in loading the results for the dataset: ", dataset, " and for the method: ", method)
                continue

            method_name=method.replace(".csv", "")
            if method_name not in methods_vs_metrics:
                methods_vs_metrics[method_name] = method_scores.copy()
            else:
                methods_vs_metrics[method_name]=pd.concat([methods_vs_metrics[method_name], method_scores.copy()],axis=1)

    # remove methods with missing datasets
    metrics_tmp=methods_vs_metrics.copy()
    for method_name in methods:
        if set(metrics_tmp[method_name]) != set(datasets):
            print(" Removing the method :", method_name, " from the results.")
            methods_vs_metrics.pop(method_name)
        print(method_name, methods_vs_metrics[method_name].shape)
    return methods_vs_metrics

def load_combined_scores(outputs_folder, methods, datasets):
    methods_vs_metrics = {}
    for method in methods:
        try:
            method_scores = pd.read_csv("{}/{}_metrics.csv".format(outputs_folder,method),index_col=0)
        except:
            print("Error in loading the combined scores for the method: ", method)
            continue
        methods_vs_metrics[method] = method_scores.copy()
    # remove methods with missing datasets
    metrics_tmp=methods_vs_metrics.copy()
    for method_name in methods:
        if set(metrics_tmp[method_name]) != set(datasets):
            print(" Removing the method :", method_name, " from the results.")
            methods_vs_metrics.pop(method_name)
        print(method_name, methods_vs_metrics[method_name].shape)
    return methods_vs_metrics

def calculate_sim_scores(adata, batch_key, label_key, batch_pairs):
    similarity_scores = {}
    for pair in tqdm((batch_pairs)):
        b1, b2 = pair.split("_")


        adata_i=adata[adata.obs[batch_key].isin([b1, b2])]
        data_b1 = adata_i[adata_i.obs[batch_key] == b1]
        data_b2 = adata_i[adata_i.obs[batch_key] == b2]

        data_b1_cts = set(data_b1.obs[label_key])
        data_b2_cts = set(data_b2.obs[label_key])

        common_cts = (data_b1_cts.intersection(data_b2_cts))

        cosine_sims = []
        for ct in common_cts:
            data_b1_ = data_b1[data_b1.obs[label_key] == ct]
            data_b2_ = data_b2[data_b2.obs[label_key] == ct]
            cosine_sims.append(np.mean(cosine_similarity(data_b1_.X, data_b2_.X)))

        similarity_scores["{}_{}".format(b1, b2)] = np.round(np.mean(cosine_sims), 2)
    return similarity_scores

def plot_metrics(overall_score, bio_scores, batch_scores, title,file_name, params, outputs_folder, xLabelsRotation = 30):
    # Define the sizes for each subplot
    sizes = [3*overall_score.shape[1], 3*bio_scores.shape[1], 3*batch_scores.shape[1]]

    # Calculate the total width
    total_width = sum(sizes)

    # Create a figure with a specified height
    fig = plt.figure(figsize=(total_width, 8))

    # Create a GridSpec with widths proportional to the specified sizes
    gs = GridSpec(1, 3, width_ratios=sizes)

    # Overall score
    ax1 = fig.add_subplot(gs[0])
    overall_score.transpose().plot(kind='bar', ax=ax1, color=params["color"], legend=False)
    ax1.set_title('Overall score', fontsize = params["fontsize"])
    ax1.set_ylabel('Score', fontsize = params["fontsize"])
    ax1.set_ylim(0, 1.)
    ax1.tick_params(axis='x', labelsize= params["fontsize"], rotation=20)
    ax1.tick_params(axis='y', labelsize= params["fontsize"])


    # Biological scores
    ax2 = fig.add_subplot(gs[1])
    bio_scores.transpose().plot(kind='bar', ax=ax2, color=params["color"],legend=False)
    ax2.set_title('Biological conservation', fontsize = params["fontsize"])
    ax2.set_ylim(0, 1.)
    ax2.tick_params(axis='x', labelsize= params["fontsize"], rotation=20)
    ax2.tick_params(axis='y', labelsize= params["fontsize"])

    # Batch scores
    ax3 = fig.add_subplot(gs[2])
    batch_scores.transpose().plot(kind='bar', ax=ax3, color=params["color"],legend=False)
    ax3.set_title('Batch correction', fontsize = params["fontsize"])
    ax3.set_ylim(0, 1.)
    ax3.tick_params(axis='x', labelsize= params["fontsize"], rotation=20)
    ax3.tick_params(axis='y', labelsize= params["fontsize"])
    # Adjust the layout
    fig.suptitle(title, fontsize = params["fontsize"]+5)
    plt.tight_layout()

    plt.show()
    fig.savefig('{}/metric_plots/{}'.format(outputs_folder, file_name), bbox_inches = 'tight')
