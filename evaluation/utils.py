import os
import warnings
import numpy as np
import seaborn as sns
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

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

def plot_metrics(overall_score, bio_scores, batch_scores, title,file_name, params, p_folder, xLabelsRotation = 30):
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
    fig.savefig('{}/{}'.format(p_folder, file_name), bbox_inches = 'tight')
