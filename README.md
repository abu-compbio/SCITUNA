<br/>
<p align="center">
    <span style="font-family: 'Courier New', Courier, monospace; font-size: 24px; color: #007ACC;">SCITUNA</span>
</p>
<h1 align="center">SCITUNA</h1>
<h2 align="center">Single-Cell data Integration Tool Using Network Alignment</h2>
<br/>

[SCITUNA](https://github.com/abu-compbio/SCITUNA/):   Single-Cell RNA-seq datasets Integration Tool Using Network Alignment with batch effect correction. Our method aligns pairs of cells from two batches and uses an iterative approach to refine the integration of each cell based on its nearest neighboring cells.

#For more information, please refer to the article which can be found at [here]([xx](https://github.com/abu-compbio/SCITUNA/)).

<br/>
<p align="center">
    <img width="100%" src="https://github.com/abu-compbio/SCITUNA/blob/main/SCITUNA.png" alt="ProtTrans Attention Visualization">
    <em>The five main stages of the SCITUNA workflow: a) preprocessing and normalization, b) dimensionality reduction and clustering,  c) construction of intra-graphs and the inter-graph, d) anchor selection, e) integration, and f) visualization of the integration results.</em>
</p>
<br/>

## Reproducibility
Below are the steps to obtain the results in the paper.

### Preparing the _SCITUNA_ environment
1. Download the repository.


2. Update the conda base\
``conda update conda -n base -y``


3. _conda-forge_ needs to be added for installations of packages.
``conda config --append channels conda-forge``


4. Create a new environment named _scituna_ with a specified Python version and install required packages.\
``conda create python=4.0.6 --name scituna --file requirements.txt -y``


5. Activate new environment.\
``conda activate scituna``


6. Adding _ipykernel_ to this new environment named _SCITUNA_.
``python -m ipykernel install --user --name scituna --display-name "SCITUNA"``

6. Install scIB package.
``pip install scib``


### Get Datasets

To download the employed datasets, follow these steps:

1. Navigate to the `data` directory:
    ```bash
    cd data
    ```

2. Run the script to download the dataset. The `dataset` argument can be either `pancreas` or `lung`:
    ```bash
    python get_data.py [dataset]
    ```

Example usage:
```bash
python get_data.py pancreas
```
### Run SCITUNA

To run SCITUNA, follow the steps provided in the Jupyter notebook `run_scituna.ipynb`.

### Analysis

To run the analysis and evaluation metrics, follow the steps provided in the Jupyter notebooks `evaluation_metrics.ipynb` and `metric_score_analysis.ipynb`.


