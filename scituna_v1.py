import rpy2
import math
import scib
import warnings
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
from rpy2 import robjects
from tqdm import tqdm as tqdm
from collections import Counter
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from rpy2.robjects import pandas2ri
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import euclidean_distances

class SciTuna:
    def __init__(self, adata, batch_key, label_key, k_neighbors=None, beta=0.5, max_iterations=10000,
                 rows_are_genes=False, verbose=False):
        self.beta = beta
        self.adata = adata
        self.verbose = verbose
        self.label_key = label_key
        self.batch_key = batch_key
        self.k_neighbors = k_neighbors
        self.max_iterations = max_iterations
        self.rows_are_genes = rows_are_genes

        self.preprocess_datasets()


    ###################################################################################################
    # 1. preprocessing the dataset
    ###################################################################################################
    def preprocess_datasets(self):
        """
        Processes an AnnData object consisting of data from two batches.

        The input AnnData object (self.adata), which contains data from two batches, is accessed globally.
        The method also uses a global boolean variable 'rows_are_genes' that indicates if rows in the matrix represent genes.

        The method produces two matrices:
            - The first matrix (reference dataset), referred to as "D_r" in our publication.
            - The second matrix (target dataset), referred to as "D_t" in our publication.
        """
        print(" - 1.Data preprocessing.")
        batches = np.unique(self.adata.obs[self.batch_key])
        self.data1_object = self.adata[self.adata.obs[self.batch_key] == batches[0]]
        data1 = self.data1_object.X.toarray()
        self.data2_object = self.adata[self.adata.obs[self.batch_key] == batches[1]]
        data2 = self.data2_object.X.toarray()



        if data1.shape[0] < data2.shape[0]:
            self.reference_dataset, self.target_dataset = data1, data2  # reference dataset - "D_r" in our publication,  target dataset - "D_t" in our publication
            self.data = pd.concat([self.data1_object.to_df(),self.data2_object.to_df()])
            self.cell_batches = pd.DataFrame(np.concatenate([self.data1_object.obs[[self.batch_key]], self.data2_object.obs[[self.batch_key]]]))[0]

        else:
            self.reference_dataset, self.target_dataset = data2, data1  # reference dataset - "D_r" in our publication,  target dataset - "D_t" in our publication
            self.data = pd.concat([self.data2_object.to_df(),self.data1_object.to_df()])
            self.cell_batches = pd.DataFrame(np.concatenate([self.data2_object.obs[[self.batch_key]], self.data1_object.obs[[self.batch_key]]]))[0]

        del data1, data2, self.data1_object, self.data2_object

        # If genes are defined as rows then we transpose the matrix
        if self.rows_are_genes:
            print("------- Data is transposed")
            self.reference_dataset = self.reference_dataset.T
            self.target_dataset = self.target_dataset.T

        self.ref_tar_data = np.concatenate((self.reference_dataset, self.target_dataset), axis=0)  # combine reference and target datasets - "D_rt" in our publication

    ###################################################################################################
    # 2. Dimensionality reduction
    ###################################################################################################
    def dimensionality_reduction(self, pca_dims=100, fresh=False):

        """
        The dimensions of both the reference and target datasets are reduced separately using
        PCA to a default of 100 dimensions or principal components (PCs). PCA is also used to
        reduce the dimension of the concatenated reference and target datasets to a 100-dimensional
        plane, by default.

        - Reducing the space of both batches combined (Inter-space), this will be used to calculate the inter-adjacency matrix.
        - Reducing the space of each dataset separately (Intra-space), this will be used to calculate the intra-adjacency matrix.

        Input:
            - ref_tar_data: the combined reference and target datasets ("D_rt" in our publication)
            - reference_dataset: the reference dataset ("D_r" in our publication)
            - target_dataset: the target dataset ("D_t" in our publication)
            - pca_dims: number of components to keep (default: 100)

        Output:
            - data_rd_1: reference dataset after inter-space transformation
            - data_rd_2: target dataset after inter-space transformation
            - S_r: reference dataset after intra-space transformation
            - S_t: target dataset after intra-space transformation
        """

        print(" - 2.Dimensionality Reduction.")
        # check if we already reduced the space, then no need to do it one more time
        if hasattr(self, 'data_rd_1') and hasattr(self, 'data_rd_2') and hasattr(self, 'S_r') and hasattr(self, 'S_t') and not fresh:
            return


        #Inter-spcae transformation
        if self.ref_tar_data.shape[1] > pca_dims:  # check that the dimension  of the original dataset is larger than pca_dims
            if pca_dims > 1:  # if n_components is larger than 1
                pca = PCA(n_components=int(pca_dims), random_state=0)
                self.S_rt = pca.fit_transform(self.ref_tar_data)  # transform the entire dataset (reference and target datasets are combined)


                #split S_rt into reference and target datasets after intra-space transformation
                self.data_rd_1 = self.S_rt[:self.reference_dataset.shape[0],:]  # reference dataset after intra-space transformation
                self.data_rd_2 = self.S_rt[self.reference_dataset.shape[0]:, :]  # target dataset after intra-space transformation

            else:
                '''
                if the n_components  is not larger than 1, then it selects the number of components such that the amount of variance
                that needs to be explained is greater than the percentage specified by n_components
                '''
                pca = PCA(pca_dims, random_state=0)
                self.S_rt = pca.fit_transform(self.ref_tar_data)
                '''
                split S_rt into reference and target datasets after intra-space transformation
                '''
                self.data_rd_1 = self.S_rt[:self.reference_dataset.shape[0], :]
                self.data_rd_2 = self.S_rt[self.reference_dataset.shape[0]:, :]


        else:
            self.data_rd_1 = self.reference_dataset
            self.data_rd_2 = self.target_dataset

        '''
        Intra-space Transformation:
            This process is similar to the one applied in Inter-Space reduction; however, it is performed separately for each batch dataset.
            The purpose is to compute the intra-adjacency matrix.
        '''
        if self.reference_dataset.shape[1] > pca_dims:
            pca1 = PCA(n_components=pca_dims, random_state=0)
            self.S_r = pca1.fit_transform(self.reference_dataset)
        else:
            self.S_r = self.reference_dataset

        if self.target_dataset.shape[1] > pca_dims:
            pca2 = PCA(n_components=pca_dims, random_state=0)
            self.S_t = pca2.fit_transform(self.target_dataset)
        else:
            self.S_t = self.target_dataset

    ####################################################################
    # Clustering
    ####################################################################
    def clustering(self, kc=30):
        """
        Description:
        The k-means algorithm is used to cluster each of the reference and target datasets,
        where k is determined using the silhouette index approach: by varying k between 2 and 30.

        Parameters:
        ----------
        kc : int
            The range of n_clusters. For example, if kc=30, k will vary between 2 and 30.
        """
        print(" - 3.Clustering.")

        '''
        SS_ref and SS_target arrays store the silhouette scores for the reference and target datasets,respectively.
        '''
        SS_ref = []
        SS_target = []
        K = range(2, kc)
        self.rkm = [-100, -100]
        for k in K:  # for k varying between 2 and 30
            self.rkm.append(KMeans(n_clusters=k, random_state=0))  # run Kmeans on the reference dataset
            self.rkm[k] = self.rkm[k].fit(self.S_r)  # store the labels for each sample in the reference data
            SS_ref.append(silhouette_score(self.S_r, self.rkm[k].labels_))  # store the silhouette score
        self.rkm = self.rkm[np.argmax(SS_ref) + 2]

        # the same procedures described above will be applied for the target dataset
        self.tkm = [-100, -100]
        for k in K:
            self.tkm.append(KMeans(n_clusters=k, random_state=0))
            self.tkm[k] = self.tkm[k].fit(self.S_t)
            SS_target.append(silhouette_score(self.S_t, self.tkm[k].labels_))  # store the inertia score
        self.tkm = self.tkm[np.argmax(SS_target) + 2]


    def inter_intra_adjacency_matrices(self):
        self.inter_adjacency_matrix()
        self.intra_dists_measurement()

    ####################################################################
    # Construct the Inter-adjacency matrix
    ####################################################################
    def inter_adjacency_matrix(self, fresh=False):
        print(" - 4.Inter-Adjacency Matrix.")
        '''
        RefNodes is an array that contains the renamed ids of each cell in the refenece dataset. Ids are named as follows r_[index of the cell in the original matrix]
        TarNodes is an array that contains the renamed ids of each cell in the target dataset. Ids are named as follows t_[index of the cell in the original matrix]
        '''
        self.ref_nodes = ['r_' + str(i) for i in list(range(self.reference_dataset.shape[0]))]
        self.tar_nodes = ['t_' + str(i) for i in list(range(self.target_dataset.shape[0]))]
        if not hasattr(self, 'interAdjMat') or fresh:
            """
            Description:
            Calculate the Euclidean distances between each reference cell r_i and all target cells.
            """
            self.inter_adj_matrix = euclidean_distances(self.data_rd_1, self.data_rd_2)

    ####################################################################
    # Construct the Intra-adjacency matrix
    ####################################################################
    def intra_dists_measurement(self, fresh=False):
        print(" - 5.Intra-Adjacency Matrix.")
        if not (hasattr(self, 'rSortedPearIdx') and hasattr(self, 'rdists') and hasattr(self,'tSortedPearIdx') and hasattr(self, 'tdists')) or fresh:
            self.tar_max_dist = -1
            self.ref_max_dist = -1

            self.ref_intra_dist = euclidean_distances(self.S_r, self.S_r)
            self.ref_intra_dist = np.triu(self.ref_intra_dist)

            self.tar_intra_dist = euclidean_distances(self.S_t, self.S_t)
            self.tar_intra_dist = np.triu(self.tar_intra_dist)

            self.tar_intra_corr = 1. - cdist(self.S_t, self.S_t, metric='correlation')
            self.ref_intra_corr = 1. - cdist(self.S_r, self.S_r, metric='correlation')

            self.tar_intra_corr[np.tril_indices_from(self.tar_intra_corr)] = -2.0
            self.ref_intra_corr[np.tril_indices_from(self.ref_intra_corr)] = -2.0
            self.tar_max_dist = np.max(self.tar_intra_dist)
            self.ref_max_dist = np.max(self.ref_intra_dist)
            ref_comb_count = int((len(self.ref_nodes) ** 2 - len(self.ref_nodes)) / 2)
            '''
            rSortedPearIDs: a data structure that holds the tuples of ref-ref cells sorted based on their pearson correlation coefficients
            '''
            self.rSortedPearIDs = [(math.floor(x / len(self.ref_nodes)), x % len(self.ref_nodes)) for x in np.argsort(-self.ref_intra_corr, axis=None)]
            self.rSortedPearIDs = self.rSortedPearIDs[:ref_comb_count]
            del self.ref_intra_corr

            '''
            tSortedPearIDs: a data structure that holds the tuples of target-target cells sorted based on their pearson correlation coefficients
            '''
            tar_comb_count = int((len(self.tar_nodes) ** 2 - len(self.tar_nodes)) / 2)
            self.tSortedPearIDs = [(math.floor(x / len(self.tar_nodes)), x % len(self.tar_nodes)) for x in np.argsort(-self.tar_intra_corr, axis=None)]
            self.tSortedPearIDs = self.tSortedPearIDs[:tar_comb_count]
            del self.tar_intra_corr


    ####################################################################
    # Graphs construction
    ####################################################################
    def graphs_construction(self, cthresh=None, skip=5):
        self.build_reference_graph(cthresh, skip)
        self.build_target_graph(cthresh, skip)
        self.check_isolated_nodes()

    def build_reference_graph(self, cthresh=None, skip=5):
        print(" - 6.Graphs construction - Gr.")
        self.graphRef = nx.DiGraph()  # reference graph is a directed graph
        self.graphRef.add_nodes_from(self.ref_nodes)  # add nodes
        if cthresh is None:
            riaep = 100 * max(20, min(list(Counter(self.rkm.labels_).values()))) / len(self.ref_nodes)
            tiaep = 100 * max(20, min(list(Counter(self.tkm.labels_).values()))) / len(self.tar_nodes)
            if riaep > 1.5 * tiaep:
                riaep /= len(np.unique(self.rkm.labels_))
            cthresh = riaep
        self.create_reference_edges(cthresh, skip)

    def create_reference_edges(self, thresh, skip):
        if skip < 100 and not hasattr(self, 'rkm'):
            self.clustering()

        edge_addition_threshold = thresh / 100 * len(self.rSortedPearIDs)  # a threshold to control the number of added edges in the graph
        # normalize edge weights
        ref_dists = []
        for (i, j) in self.rSortedPearIDs:
            ref_dists.append(self.ref_intra_dist[i, j])
        self.max_ref_dist = np.max(ref_dists)
        self.min_ref_dist = np.min(ref_dists)
        del ref_dists
        e = 0
        while e < edge_addition_threshold:
            (i, j) = self.rSortedPearIDs[e]  # get ref-ref ids sorted based on the pearson corr.
            # normalize the edge weight (min-max normalization)
            '''
            Note that for k edges, we don't check their labels (after clustering),
            where k =skip/100 * edge_addition_threshold
            '''

            dst = (self.ref_intra_dist[i, j] - self.min_ref_dist) / (self.max_ref_dist - self.min_ref_dist)
            if e > int(skip / 100 * edge_addition_threshold):
                if self.rkm.labels_[i] == self.rkm.labels_[ j]:  # check if both reference cells were assigned the same cluster

                    # initially the graph is bidirectional, a->b and b->a
                    self.graphRef.add_edge('r_' + str(i), 'r_' + str(j), dist=dst)  # add a->b
                    self.graphRef.add_edge('r_' + str(j), 'r_' + str(i), dist=dst)  # add b->a

            else:

                self.graphRef.add_edge('r_' + str(i), 'r_' + str(j), dist=dst)  # add a->b
                self.graphRef.add_edge('r_' + str(j), 'r_' + str(i), dist=dst)  # add b->a

            e += 1

    # the same as reference graph
    def build_target_graph(self, cthresh=None, skip=5):
        print(" - 7.Graph construction - Gt.")
        self.graphTar = nx.DiGraph()
        self.graphTar.add_nodes_from(self.tar_nodes)
        if cthresh is None:
            cthresh = 100 * max(20, min(list(Counter(self.tkm.labels_).values()))) / len(self.tar_nodes)
        self.add_target_edges(cthresh, skip)

    def add_target_edges(self, thresh, skip):
        if skip < 100 and not hasattr(self, 'tkm'):
            self.findNClusters()

        edge_addition_threshold = thresh / 100 * len(self.tSortedPearIDs)

        # normalize edge weights
        tar_dists = []
        for (i, j) in self.tSortedPearIDs:
            tar_dists.append(self.tar_intra_dist[i, j])
        self.max_tar_dist = np.max(tar_dists)
        self.min_tar_dist = np.min(tar_dists)
        e = 0
        while e < edge_addition_threshold:
            (i, j) = self.tSortedPearIDs[e]
            dst = (self.tar_intra_dist[i, j] - self.min_tar_dist) / (self.max_tar_dist - self.min_tar_dist)
            if e > int(skip / 100 * edge_addition_threshold):
                if self.tkm.labels_[i] == self.tkm.labels_[j]:
                    self.graphTar.add_edge('t_' + str(i), 't_' + str(j), dist=dst)
                    self.graphTar.add_edge('t_' + str(j), 't_' + str(i), dist=dst)
            else:

                self.graphTar.add_edge('t_' + str(i), 't_' + str(j), dist=dst)
                self.graphTar.add_edge('t_' + str(j), 't_' + str(i), dist=dst)
            e += 1

    ####################################################################
    # Handling isolated nodes
    ####################################################################
    def check_isolated_nodes(self):
        print(" - 8.Graph construction - Check for Isolated Nodes.")
        ref_degrees = [deg for (node, deg) in self.graphRef.degree() if deg != 0]  # get the degree of nodes in the reference graph
        min_ref_degree = int(np.min(ref_degrees))

        isolated_nodes = list(nx.isolates(self.graphRef)).copy()  # list of all isolated nodes
        for r in isolated_nodes:  # list all isolated nodes in reference graph

            ref_node = int(r[2:])  # retrieve the id of the isolated node

            k = 0  # to control the number of added edges
            # iterate over the ref-ref tuples that are sorted based on the pearson correlation coefficients
            for e in range(len(self.rSortedPearIDs)):  # check for pairs sorted based on the pearson distance
                (i, j) = self.rSortedPearIDs[e]
                # retrieve the distance

                if ref_node == i:  # check if the isolated node is the first item in the tuple, if so, add an edge ref_i->j
                    dst = (self.ref_intra_dist[i, j] - self.min_ref_dist) / (self.max_ref_dist - self.min_ref_dist)  # normalize the distance
                    self.graphRef.add_edge('r_' + str(i), 'r_' + str(j), dist=dst)
                    k += 1
                elif ref_node == j:  # if the isolated node is j, then add an edge ref_j->i
                    dst = (self.ref_intra_dist[i, j] - self.min_ref_dist) / (self.max_ref_dist - self.min_ref_dist)  # normalize the distance
                    self.graphRef.add_edge('r_' + str(j), 'r_' + str(i), dist=dst)
                    k += 1
                if k == min_ref_degree:  # it stops if we add k edges
                    break

        tar_degrees = [deg for (node, deg) in self.graphTar.degree() if deg != 0]  # get the degree of nodes in the target graph
        min_Tar_degree = int(np.min(tar_degrees))
        isolated_nodes = list(nx.isolates(self.graphTar)).copy()  # list all isolated nodes
        for t in isolated_nodes:  # list all isolated nodes in the target graph

            tar_node = int(t[2:])  # retrieve the id of node
            k = 0  # to control the number of added edges
            # iterate over the tar-tar tuples that are sorted based on the pearson correlation coefficients
            for e in range(len(self.tSortedPearIDs)):  # check for pairs sorted based on the pearson distance
                (i, j) = self.tSortedPearIDs[e]
                # retrieve the distance
                if tar_node == i:  # check if the isolated node is the first item in the tuple, if so, add an edge ref_i->j
                    dst = (self.tar_intra_dist[i, j] - self.min_tar_dist) / (self.max_tar_dist - self.min_tar_dist)  # normalize the distance
                    self.graphTar.add_edge('t_' + str(i), 't_' + str(j), dist=dst)
                    k += 1
                elif tar_node == j:  # if the isolated node is j, then add an edge ref_j->i
                    dst = (self.tar_intra_dist[i, j] - self.min_tar_dist) / (self.max_tar_dist - self.min_tar_dist)  # normalize the distance
                    self.graphTar.add_edge('t_' + str(j), 't_' + str(i), dist=dst)
                    k += 1
                if k == min_Tar_degree:  # if we added k edges, then stop
                    break

    ####################################################################
    # Anchors selection
    ####################################################################
    def anchors_selection(self):
        """
        INPUT:
            - data: Gene expression data as a DataFrame. The first k rows correspond to the reference dataset, and the remaining rows correspond to the target dataset.
            - HVG: List of highly variable genes.
        OUTPUT:
            A dictionary containing reference cells as keys and their matched target cells as values.
        Seurat Anchors:
        - CreateSeuratObject: Create a Seurat object from a feature (e.g., gene) expression matrix. The expected format of the input matrix is features x cells.
        - FindVariableFeatures: Identifies highly variable genes. (More info: https://search.r-project.org/CRAN/refmans/Seurat/html/FindVariableFeatures.html)
        - FindIntegrationAnchors: Find a set of anchors between a list of Seurat objects.
        """
        print(" - 9.Anchors Selection")



        pandas2ri.activate()
        robjects.globalenv['data'] = robjects.conversion.py2rpy(self.data)
        robjects.globalenv['batches'] = robjects.conversion.py2rpy(self.cell_batches)
        r_output = robjects.r(
        '''
        library ("Seurat")
        #options(future.globals.maxSize = 1000 * 1024^2)
        seurat_obj <- CreateSeuratObject(t(data))
        seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", verbose = F,nfeatures = 2000)

        seurat_obj$Batch <- batches
        data.list <- SplitObject(seurat_obj, split.by = "Batch") #split by batch

        #get anchors
        anchors_dataframe <- FindIntegrationAnchors(object.list = data.list,dims = 1:30, verbose = F )
        anchors_dataframe=anchors_dataframe@anchors
        output=list(anchors_dataframe)
        '''
        )
        anchors_dataframe = r_output[0]  # Seurat anchors
        anchors_dataframe = anchors_dataframe.astype({'cell1': int, 'cell2': int})  # convert columns' data types
        anchors_dataframe = anchors_dataframe[anchors_dataframe["dataset1"] == 1]  # select only dataset 1 rows - reference dataset
        anchors_dataframe = anchors_dataframe[anchors_dataframe.score > 0.0]  # remove rows  with score 0

        # initialize anchhor dataset
        self.ref_tar_matchings = []  # matched pairs
        ref_cells_anchors = np.unique(anchors_dataframe.cell1)
        for idx in ref_cells_anchors:
            r = idx - 1  # select reference cell R_i
            rMatchedTarCells = np.array(
                anchors_dataframe[anchors_dataframe.cell1 == idx]["cell2"]) - 1  ##select its matched target T_j
            dists = {}
            for t in rMatchedTarCells:
                dists[t] = self.inter_adj_matrix[r, t]
            sorted_anchors = sorted(dists.items(), key=lambda item: item[1])
            self.ref_tar_matchings.append((r, sorted_anchors[0][0]))  # match Ri and Tj

        self.ref_tar_matchingsD = {}
        for i in self.ref_tar_matchings:
            self.ref_tar_matchingsD[i[0]] = [i[1]]
        self.get_anchors()

    def get_anchors(self):
        self.anchors_data = dict()
        for m in self.ref_tar_matchings:
            xx = self.reference_dataset[m[0], :]
            yy = self.target_dataset[m[1], :]
            if m[0] not in self.anchors_data:
                self.anchors_data[m[0]] = []
            self.anchors_data[m[0]].append((xx, yy, m[1]))

    ####################################################################
    # Integration
    ####################################################################
    '''
    1. intvectors: A data structure to store the integration vectors.
    2. alphas: A data structure to store the alpha values for all nodes in the reference graph.
       The data structure is defined as follows:
       For example, the alpha values of reference cell ref_0 and its neighbors will be:
       {
           0: [
               0.1xxx -> alpha 0, (only if reference cell is matched with a target cell)
               0.3 -> alpha of a neighbor of ref_0 (again matched with a target cell)
               0.1 -> alpha of another neighbor of ref_0 (again matched with a target cell)
               ...
           ]
       }
    3. alphas_na: A data structure to store the alpha values for all nodes in the reference graph that are not matched with any target cell.
       The data structure is defined as follows:
       For example, the alpha values of reference cell 1 and its neighbors will be:
       {
           0: [
               0.3 -> alpha of a neighbor of ref_1 (again not matched with a target cell)
               0.2 -> alpha of another neighbor of ref_1 (again not matched with a target cell)
               ...
           ]
       }
       Note that the sum of alpha values in alphas and alphas_na for the same reference cell must be equal to 1.

    4. closests: A data structure that stores the IDs of the neighbors of reference cell x that are matched with a target cell.
    5. closests_na: A data structure that stores the IDs of the neighbors of reference cell x that are not matched with a target cell.
    6. rcdists: A data structure that stores the distances to the neighbors of reference cell x that are matched with a target cell.
    7. rcdists_na: A data structure that stores the distances to the neighbors of reference cell x that are not matched with a target cell.
    8. tcdists: A data structure that holds the distances between the anchor of reference cell x and the anchors of its neighbors.
       Note that if there is no such an edge anchor_ref_i and anchor_neighbor_i in the target graph, then we assign a value as the maximum distance.

    '''

    def integrate_datasets(self):
        print(" - 10.Integrating Datasets.")

        self.int_vectors = []  # integration vectors
        self.alphas = {}  # alpha values for nodes with anchors
        self.alphas_na = {}  # alpha values for nodes without anchors

        self.r_neighbors_woa = {}  # the neighbor (without anchor) of reference cell
        self.r_neighbors_wa = {}

        self.r_neighbors_distances_woa = {}
        self.r_neighbors_distances_wa = {}


        self.r_a_distances = {}
        self.rAnchor_nAnchor_distances = {}

        if self.k_neighbors is None:
            self.k_neighbors = max(20, min(30, min(list(Counter(self.rkm.labels_).values())) - 1))

        for i in range(self.reference_dataset.shape[0]):
            self.r_neighbors_wa[i], self.r_neighbors_distances_wa[i], self.rAnchor_nAnchor_distances[i], self.r_a_distances[i], self.r_neighbors_woa[i], self.r_neighbors_distances_woa[
                i] = self.initializations("r_" + str(i))


        # calculate the integration vector
        for i in range(self.reference_dataset.shape[0]):
            self.set_int_vectors(i)

        self.prev_int_vectors = np.array(self.int_vectors)
        self.curr_int_vectors = np.zeros_like(self.prev_int_vectors)
        self.diff = [1000, np.sum(self.curr_int_vectors)]
        self.iteration = 0
        while abs(self.diff[-2] - self.diff[-1]) / (
                abs(self.diff[-1]) + 0.00001) > 0.00001 and self.iteration < self.max_iterations:
            self.iteration += 1
            for i in range(self.reference_dataset.shape[0]):
                if i in self.anchors_data:
                    self.curr_int_vectors[i] = self.int_vectors[i]
                    continue
                self.curr_int_vectors[i] = self.set_curr_int_vectors(i, self.iteration)
            self.diff.append(np.sum(self.curr_int_vectors))
            self.prev_int_vectors = np.copy(self.curr_int_vectors)
        self.correct_reference_dataset()
        self.integrated_data = np.vstack([self.corrected_reference_data, self.target_dataset])
        print(" - 11.Done.")

    def get_cell_anchor_lengths(self, r_i):
        # r_i is the reference cell
        lengths = {}  # data structure that stores the anchor of r_i and the distance between them in the inter-adja matrix
        # if the reference cell was not matched with an anchor, then return an empty dictinoary
        if r_i not in self.ref_tar_matchingsD:
            return lengths

        for t_j in self.ref_tar_matchingsD[r_i]:
            # normalize the distance in the inter-adj matris
            lengths[t_j] = (self.inter_adj_matrix[r_i, t_j] - np.min(self.inter_adj_matrix)) / (
                    np.max(self.inter_adj_matrix) - np.min(self.inter_adj_matrix))
        return lengths

    def initializations(self, r_i):
        def take_second(elem):
            return elem[1]

        r_i_neighbors = set()  # a set that will store the neighbors of the the refernce cell node r_i

        for n_j in self.graphRef.neighbors(r_i):
            r_i_neighbors.add((n_j, self.graphRef.edges[r_i, n_j]['dist']))

        # sort neighbors based on the distance
        r_i_neighbors = sorted(r_i_neighbors, key=take_second)

        n_threshold = self.k_neighbors
        ri_neighbors_woa = []  # neighbors of "node" that are not matched a target
        ri_neighbors_distances_woa = []  # distances to the neighbors of "node" that are not matched a target
        ri_neighbors_wa = []  # ids of neighbors of "node" that are  matched a target
        ri_neighbors_distances_wa = []  # distances to the neighbors of "node" that are matched a target
        riAnchor_nAnchor_distances = []  # distances between anchors of "node" and anchors of its neighbors
        ri_a_distances = []

        if int(r_i[2:]) in self.anchors_data:  # checking if the reference cell was matched with any target
            anchor_of_ri = 't_' + str(self.anchors_data[int(r_i[2:])][0][2])
        else:
            anchor_of_ri = 'MISSING Anchor'

        if int(r_i[2:]) in self.anchors_data:  # if the reference cell "r_i" was matched with a target cell
            ri_neighbors_wa.append(int(r_i[2:]))
            ri_neighbors_distances_wa.append(0)
            riAnchor_nAnchor_distances.append(0)
            ri_a_distances.append(self.get_cell_anchor_lengths(int(r_i[2:])))

        for i in range(n_threshold):  # for each neighbor of "node"
            if i < len(r_i_neighbors):

                anchor_of_ni = self.get_cell_anchor_lengths(
                    int(r_i_neighbors[i][0][2:]))  # get the anchor of its neighbor
                if len(anchor_of_ni) == 0:  # if no anchor selected, then skip
                    ri_neighbors_woa.append(int(r_i_neighbors[i][0][2:]))  # add the neighbor id to ri_neighbors_woa
                    ri_neighbors_distances_woa.append(
                        r_i_neighbors[i][1])  # add the  distance bwtween "node" and its neighbor  to r_r_disances.
                else:
                    # if the neighbor was matched with an anchor
                    ri_neighbors_wa.append(int(r_i_neighbors[i][0][2:]))  # add the id of the neighbor to closests
                    ri_neighbors_distances_wa.append(r_i_neighbors[i][1])  # add the distance between "node" and its neighbor
                    rn = r_i_neighbors[i][0]  # id of the neighbor
                    anchor_of_ni = 't_' + str(self.anchors_data[int(rn[2:])][0][2])  # get the anchor of its neighbor
                    if anchor_of_ri == anchor_of_ni:  # if both "r_i" and its neighbor j were matched with the same anchor then the distance must be 0
                        riAnchor_nAnchor_distances.append(0)
                    elif (anchor_of_ri,
                          anchor_of_ni) in self.graphTar.edges():  # if not, then check if there is an edge between them in the target graph and get its distance
                        riAnchor_nAnchor_distances.append(self.graphTar.edges[anchor_of_ri, anchor_of_ni]['dist'])
                    else:  # if there is no such an edge, or "r_i" was not matched with an anchor, then append the maximum distance in the inter-adj- matrix
                        riAnchor_nAnchor_distances.append(
                            (self.tar_max_dist - np.min(self.tar_intra_dist)) / (np.max(self.tar_intra_dist) - np.min(self.tar_intra_dist)))
                    ri_a_distances.append(
                        self.get_cell_anchor_lengths(int(r_i_neighbors[i][0][2:])))  # add the id of its anchor
        return ri_neighbors_wa, ri_neighbors_distances_wa, riAnchor_nAnchor_distances, ri_a_distances, ri_neighbors_woa, ri_neighbors_distances_woa

    def get_alphas(self, r_i):
        #Eucledian distances between r_i and it's closest neighbors
        dists = list((np.array(self.r_neighbors_distances_wa[r_i]) + np.array(self.rAnchor_nAnchor_distances[r_i])) / 2)
        dists_na = list(np.array(self.r_neighbors_distances_woa[r_i]))

        # anchor lengths of r_i and it's closest neighbors
        riAnchor_nAnchor = self.r_a_distances[r_i]

        alphas = []
        alphas_na = []

        for i in range(len(dists)):
            v = self.beta * math.exp(-dists[i]) + (1 - self.beta) * math.exp(- list(riAnchor_nAnchor[i].values())[0])
            alphas.append(v)

        for i in range(len(dists_na)):
            v = self.beta * math.exp(-dists_na[i])
            alphas_na.append(v)
        sum_alphas = np.sum(alphas) + np.sum(alphas_na)

        alphas = [[x / sum_alphas] for x in alphas]
        alphas_na = [[x / sum_alphas] for x in alphas_na]

        return alphas, alphas_na

    def set_int_vectors(self, r_i):
        self.alphas[r_i], self.alphas_na[r_i] = self.get_alphas(r_i)
        int_vector = np.zeros(len(self.reference_dataset[0]))
        for v, w in zip(self.alphas[r_i], self.r_neighbors_wa[r_i]):
            for i, y in zip(range(len(self.anchors_data[w])), v):
                int_vector += y * (self.anchors_data[w][i][1] - self.anchors_data[w][i][0])
        self.int_vectors.append(int_vector)

    def set_curr_int_vectors(self, r_i, iteration):
        int_vector = np.zeros(len(self.reference_dataset[0]))
        for v, w in zip(self.alphas[r_i], self.r_neighbors_wa[r_i]):
            int_vector += v * (self.prev_int_vectors[w])

        if iteration != 1:  # if not the first iteration, then include all neighbors that are not matched with an anchor,
            for v, w in zip(self.alphas_na[r_i], self.r_neighbors_woa[r_i]):
                int_vector += v * (self.prev_int_vectors[w])

        return int_vector

    def correct_reference_dataset(self):
        self.corrected_reference_data = self.reference_dataset.copy().astype('float64')
        for i in range(self.corrected_reference_data.shape[0]):
            self.corrected_reference_data[i, :] += self.curr_int_vectors[i]
