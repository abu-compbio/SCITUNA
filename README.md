<br/>
<h1 align="center">SCITUNA</h1>
<h2 align="center">Single-Cell data Integration Tool Using Network Alignment</h2>
<br/>

[SCITUNA](https://github.com/abu-compbio/SCITUNA/): a novel single-cell data integration approach that combines both _graph-based_ and _anchor-based_ techniques. SCITUNA constructs a graph for each batch to represent intra-batch cell similarities, and a bipartite graph to capture inter-batch similarities. This transforms the integration problem into a many-to-one matching problem, where cells from a query batch are matched with cells from a reference batch. The resulting matches are then used to transform the query cell space to the reference cell space.

- SCITUNA operates directly in the original gene expression space.
- The method introduces a novel batch ordering strategy based on optimal transport cost.

#For more information, please refer to our article: https://doi.org/10.1186/s12859-025-06087-3

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
 




## Install SCITUNA
These steps will help you set up the SCITUNA environment and install the necessary dependencies in both Python and R.



#### 1. Create a virtual environment
To ensure reproducibility and avoid conflicts with other packages, it is recommended to use a separate Conda environment for SCITUNA.

```bat
conda create -n SCITUNA python=3.10
conda activate SCITUNA
```
This creates and activates a new environment named SCITUNA with Python 3.10. You can choose a different version if needed, but compatibility with required packages is tested for Python 3.10.

#### 2. Install Python Dependencies
Once the environment is active, install the required Python packages listed in requirements.txt using pip.

Make sure requirements.txt is in your current directory.

```bat
pip install -r requirements.txt
```
#### 3. Install Seurat in R
SCITUNA also leverages functionality from the R package Seurat, which is widely used for single-cell RNA-seq data analysis.

We recommend installing Seurat version 3, as SCITUNA was developed and tested using this version to ensure compatibility and reproducibility.


## Additional data
We provide t-SNE and UMAP plots for a deeper analysis of the results. You can access them through this [Google Drive link](https://drive.google.com/drive/folders/1WnwBQritr3vc0CYv05CyzSkAdJrJzlgc?usp=drive_link).


## Citation
Houdjedj, A., Marouf, Y., Myradov, M. et al. SCITUNA: single-cell data integration tool using network alignment. BMC Bioinformatics 26, 92 (2025). https://doi.org/10.1186/s12859-025-06087-3
