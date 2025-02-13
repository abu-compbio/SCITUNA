<br/>
<h1 align="center">SCITUNA</h1>
<h2 align="center">**S**ingle-**C**ell data **I**ntegration **T**ool **U**sing **N**etwork **A**lignment</h2>
<br/>

[SCITUNA](https://github.com/abu-compbio/SCITUNA/): a novel single-cell data integration approach that combines both _graph-based_ and _anchor-based_ techniques. SCITUNA constructs a graph for each batch to represent intra-batch cell similarities, and a bipartite graph to capture inter-batch similarities. This transforms the integration problem into a many-to-one matching problem, where cells from a query batch are matched with cells from a reference batch. The resulting matches are then used to transform the query cell space to the reference cell space.

- SCITUNA operates directly in the original gene expression space.
- The method introduces a novel batch ordering strategy based on optimal transport cost.

#For more information, please refer to the article which can be found at [here]([xx](https://github.com/abu-compbio/SCITUNA/)).

<br/>
<p align="center">
    <img width="100%" src="https://github.com/abu-compbio/SCITUNA/blob/main/SCITUNA.png" alt="ProtTrans Attention Visualization">
    <em>The five main stages of the SCITUNA workflow: a) preprocessing and normalization, b) dimensionality reduction and clustering,  c) construction of intra-graphs and the inter-graph, d) anchor selection, e) integration, and f) visualization of the integration results.</em>
</p>
<br/>

## Run SCITUNA
Below are the steps to obtain the results in the paper.


### Get Datasets

To download the employed datasets, follow these steps:

1. Navigate to the `data` directory:
    ```bash
    cd data
    ```

2. Run the script to download the dataset. The `dataset` argument can be either `pancreas`, `lung`, `small_atac_peaks` or `small_atac_windows`:
    ```bash
    python get_data.py [dataset]
    ```

Example usage:
```bash
python get_data.py pancreas
```
### Multi-batch Integrations

To integrate multiple batches using **SCITUNA**, run the following command:  

 ```bash
python multi_batch_integration.py --i [input_dataset] --b [batch_id] --c [num_cores]
 ```
 **Arguments**
 
--i (input_dataset): The dataset file located in **"data/"** (supported formats: H5AD).

--b (batch_id): The column name in **".obs"** that indicates batch labels for integration.

--c (num_cores): Number of CPU cores to use for parallel processing.


### Pairwise Integrations

To perform **pairwise batch integration** using **SCITUNA**, run the following command:  

 ```bash
python pairwise_integration.py --i [input_dataset] --b [batch_id] --c [num_cores]
 ```
 

We provide t-SNE and UMAP plots for a deeper analysis of the results. You can access them through this [Google Drive link](https://drive.google.com/drive/folders/1WnwBQritr3vc0CYv05CyzSkAdJrJzlgc?usp=drive_link).



