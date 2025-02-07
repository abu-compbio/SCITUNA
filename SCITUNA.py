import gc
import math
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import sys
from rpy2 import robjects
from collections import Counter
from sklearn.cluster import KMeans
from memory_profiler import profile
from rpy2.robjects import pandas2ri
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")


class SCITUNA:
    def __init__(self, adata,
                 batch_key,
                 o_dir,
                 k_neighbors=None,
                 beta=0.5,
                 max_iterations=10000):
        self.beta = beta
        self.adata = adata
        self.batch_key = batch_key
        self.k_neighbors = k_neighbors
        self.max_iterations = max_iterations
        self.o_dir = o_dir
        self.preprocessing()


    def preprocessing(self):
        print("\tData Preprocessing.")
        batches = sorted(Counter(self.adata.obs[self.batch_key]).items(), key=lambda x: x[1], reverse=False)

        self.D_q = self.adata[self.adata.obs[self.batch_key] == batches[0][0]].to_df()
        self.D_r = self.adata[self.adata.obs[self.batch_key] == batches[1][0]].to_df()



        self.cell_ids = np.concatenate([self.D_q.index, self.D_r.index])
        self.gene_ids = self.D_q.columns
        self.batch_ids = pd.DataFrame(np.concatenate([self.adata[self.D_q.index].obs[[self.batch_key]],
                                                      self.adata[self.D_r.index].obs[[self.batch_key]]]))[0]

    def reduce_dimensions(self, pca_dims=100):
        print("\tDimensionality Red.")

        if len(self.gene_ids) > pca_dims:
            pca = PCA(n_components=int(pca_dims), random_state=0)
            self.S_qr = pca.fit_transform(np.concatenate([self.D_q, self.D_r], axis=0))
            self.S_q =  pca.fit_transform(self.D_q)
            self.S_r =  pca.fit_transform(self.D_r)


        else:
            raise ValueError(
                f"Invalid PCA configuration: The number of dimensions in the data "
                f"must be greater than or equal to the number of components chosen for PCA ({pca_dims})."
            )

    def inter_intra_similarities(self):
        print("\tInter-Similarities. ")
        self.compute_inter_similarities()

        print("\tIntra-Similarities.")
        self.compute_intra_similarities()

    def compute_inter_similarities(self):
        self.q_nodes = ['q_' + str(i) for i in list(range(self.D_q.shape[0]))]
        self.r_nodes = ['r_' + str(i) for i in list(range(self.D_r.shape[0]))]
        self.inter_dists = euclidean_distances(self.S_qr[:len(self.D_q)], self.S_qr[len(self.D_q):])

    def compute_intra_similarities(self, fresh=False):
        self.query_intra_dist = euclidean_distances(self.S_q, self.S_q)
        self.ref_intra_dist =   euclidean_distances(self.S_r, self.S_r)
        self.ref_intra_corr =   1. - cdist(self.S_r, self.S_r, metric='correlation')
        self.query_intra_corr = 1. - cdist(self.S_q, self.S_q, metric='correlation')
        # self.ref_intra_corr =   np.corrcoef(self.S_r)
        # self.query_intra_corr = np.corrcoef(self.S_q)

    def construct_edges(self):

        self.ref_intra_corr[np.tril_indices_from(self.ref_intra_corr)] = -2.0
        self.query_intra_corr[np.tril_indices_from(self.query_intra_corr)] = -2.0

        self.r_max_dist = np.max(self.ref_intra_dist)
        self.q_max_dist = np.max(self.query_intra_dist)

        self.r_min_dist = np.min(self.ref_intra_dist[~np.eye(self.ref_intra_dist.shape[0], dtype=bool)])
        self.q_min_dist = np.min(self.query_intra_dist[~np.eye(self.query_intra_dist.shape[0], dtype=bool)])

        # self.ref_intra_dist = np.triu(self.ref_intra_dist)
        # self.query_intra_dist = np.triu(self.query_intra_dist)

        p_q = 100 * max(20, min(list(Counter(self.qkm.labels_).values()))) / len(self.q_nodes)
        p_r = 100 * max(20, min(list(Counter(self.rkm.labels_).values()))) / len(self.r_nodes)
        if p_q > 1.5 * p_r:
            p_q /= len(np.unique(self.qkm.labels_))
        self.M_q = int(np.ceil(p_q / 100 * (int((len(self.q_nodes) ** 2 - len(self.q_nodes)) / 2))))
        self.M_r = int(np.ceil(p_r / 100 * (int((len(self.r_nodes) ** 2 - len(self.r_nodes)) / 2))))

        # Query
        triu_rows, triu_cols = np.triu_indices(self.query_intra_corr.shape[0], k=1)
        top_k_indices = np.argpartition(-self.query_intra_corr[triu_rows, triu_cols], self.M_q)[:self.M_q]
        sorted_order = np.argsort(-self.query_intra_corr[triu_rows, triu_cols][top_k_indices])
        self.q_edges = np.vstack((triu_rows[top_k_indices][sorted_order], triu_cols[top_k_indices][sorted_order])).T

        # Reference
        triu_rows, triu_cols = np.triu_indices(self.ref_intra_corr.shape[0], k=1)
        top_k_indices = np.argpartition(-self.ref_intra_corr[triu_rows, triu_cols], self.M_r)[:self.M_r]
        sorted_order = np.argsort(-self.ref_intra_corr[triu_rows, triu_cols][top_k_indices])
        self.r_edges = np.vstack((triu_rows[top_k_indices][sorted_order], triu_cols[top_k_indices][sorted_order])).T


    def clustering(self, kc=30):
        print("\tClustering.")
        K = range(2, kc)
        def query_kmeans(k):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(self.S_q)
            score = silhouette_score(self.query_intra_dist, kmeans.labels_, metric='precomputed')
            # score = silhouette_score(self.S_q, kmeans.labels_)
            return k, kmeans, score

        def reference_kmeans(k):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(self.S_r)
            score = silhouette_score(self.ref_intra_dist, kmeans.labels_, metric='precomputed')
            # score = silhouette_score(self.S_r, kmeans.labels_)
            return k, kmeans, score

        query_silhouette_scores = []
        ref_silhouette_scores = []

        # Query
        self.qkm = [-100, -100]
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(query_kmeans, K))
        for k, kmeans, score in results:
            self.qkm.append(kmeans)
            query_silhouette_scores.append(score)
        self.qkm = self.qkm[np.argmax(query_silhouette_scores) + 2]

        # Reference
        self.rkm = [-100, -100]
        with ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(reference_kmeans, K))
        for k, kmeans, score in results:
            self.rkm.append(kmeans)
            ref_silhouette_scores.append(score)
        self.rkm = self.rkm[np.argmax(ref_silhouette_scores) + 2]

        del self.S_qr, self.S_r, self.S_q
        gc.collect()


    def build_graphs(self, skip=5):

        print("\tCons. Graphs")

        self.Gq = nx.DiGraph()
        self.Gr = nx.DiGraph()
        self.Gq.add_nodes_from(self.q_nodes)  # add nodes
        self.Gr.add_nodes_from(self.r_nodes)  # add nodes


        print("\t\tBuild Gq")
        # Query
        M = self.M_q
        S = int(skip / 100 * M) + 1
        for (i, j) in self.q_edges[:S]:
            dst = ((self.query_intra_dist[i, j] - self.q_min_dist) /
                   (self.q_max_dist - self.q_min_dist))
            self.Gq.add_edge('q_' + str(i), 'q_' + str(j), dist=dst)  # add a->b
            self.Gq.add_edge('q_' + str(j), 'q_' + str(i), dist=dst)  # add b->a

        for (i, j) in self.q_edges[S:M]:
            if self.qkm.labels_[i] == self.qkm.labels_[j]:
                dst = ((self.query_intra_dist[i, j] - self.q_min_dist) /
                       (self.q_max_dist - self.q_min_dist))
                self.Gq.add_edge('q_' + str(i), 'q_' + str(j), dist=dst)  # add a->b
                self.Gq.add_edge('q_' + str(j), 'q_' + str(i), dist=dst)  # add b->a
        del self.q_edges
        gc.collect()

        print("\t\tBuild Gr")
        # Ref
        M = self.M_r
        S = int(skip / 100 * M) + 1
        self.rn = set()
        self.re = []
        for (i, j) in self.r_edges[:S]:
            self.re.append(i)
            self.re.append(j)
            if i not in self.matched_ref or j not in self.matched_ref:
                continue
            else:
                dst = ((self.ref_intra_dist[i, j] - self.r_min_dist) /
                       (self.r_max_dist - self.r_min_dist))
                self.Gr.add_edge('r_' + str(i), 'r_' + str(j), dist=dst)  # add a->b
                self.Gr.add_edge('r_' + str(j), 'r_' + str(i), dist=dst)  # add b->a

        for (i, j) in self.r_edges[S:M]:
            if self.rkm.labels_[i] == self.rkm.labels_[j]:
                self.re.append(i)
                self.re.append(j)
                if (i not in self.matched_ref or
                    j not in self.matched_ref):
                    continue

                else:
                    dst = ((self.ref_intra_dist[i, j] - self.r_min_dist) /
                           (self.r_max_dist - self.r_min_dist))
                    self.Gr.add_edge('r_' + str(i), 'r_' + str(j), dist=dst)  # add a->b
                    self.Gr.add_edge('r_' + str(j), 'r_' + str(i), dist=dst)  # add b->a

        degrees = np.bincount(self.re)
        self.min_ref_degree = 2 * np.min(degrees[degrees > 0])


        del degrees, self.r_edges
        gc.collect()
        self.connect_isolated_nodes()


    def connect_isolated_nodes(self):
        query_degrees = [deg for (node, deg) in self.Gq.degree() if
                       deg != 0]  # get the degree of nodes in the reference graph

        min_query_degree = int(np.min(query_degrees))
        isolated_nodes = [int(i[2:]) for i in list(nx.isolates(self.Gq))]
        for k in isolated_nodes:
            row_values = [(k, j, -self.query_intra_corr[k, j]) for j in range(k+1,len(self.query_intra_corr))]
            col_values = [(k, i, -self.query_intra_corr[i, k]) for i in range(k)]
            edges = sorted(row_values + col_values, key=lambda x: x[2])[:min_query_degree]
            for edge in edges:
                dst = ((self.query_intra_dist[edge[0], edge[1]] - self.q_min_dist) /
                                       (self.q_max_dist - self.q_min_dist))
                self.Gq.add_edge('q_' + str(edge[0]), 'q_' + str(edge[1]), dist=dst)


        del self.query_intra_corr
        gc.collect()

        isolated_nodes = list(nx.isolates(self.Gr)).copy()  # list all isolated nodes
        self.re = np.unique(self.re)
        isolated_nodes = [int(i[2:]) for i in list(nx.isolates(self.Gr))
                           if int(i[2:]) not in self.re
                           and int(i[2:]) in self.matched_ref
                           ]

        for k in isolated_nodes:
            row_values = [(k, j, -self.ref_intra_corr[k, j]) for j in range(k+1, len(self.ref_intra_corr))]
            col_values = [(k, i, -self.ref_intra_corr[i, k]) for i in range(k)]
            edges = sorted(row_values + col_values, key=lambda x: x[2])[:self.min_ref_degree]
            for edge in edges:
                dst = ((self.ref_intra_dist[edge[0], edge[1]] - self.r_min_dist) /
                                     (self.r_max_dist - self.r_min_dist))
                self.Gr.add_edge('r_' + str(edge[0]), 'r_' + str(edge[1]), dist=dst)
        del self.ref_intra_corr
        gc.collect()


    def anchors_selection(self):

        print("\tAnchors Selection")

        pandas2ri.activate()
        robjects.globalenv['data'] = robjects.conversion.py2rpy(pd.concat([self.D_q,
                                                                           self.D_r]))

        robjects.globalenv['batches'] = robjects.conversion.py2rpy(pd.DataFrame(
            np.concatenate([self.adata[self.D_q.index].obs[[self.batch_key]],
                            self.adata[self.D_r.index].obs[[self.batch_key]]]))[0])
        r_output = robjects.r(
            '''
            library ("Seurat")
            options(future.globals.maxSize = 10000 * 1024^2)
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

        del robjects.globalenv['data']
        del robjects.globalenv['batches']
        gc.collect()

        anchors_dataframe = r_output[0]  # Seurat anchors
        anchors_dataframe = anchors_dataframe.astype({'cell1': int,
                                                      'cell2': int})

        anchors_dataframe = anchors_dataframe[anchors_dataframe["dataset1"] == 1]
        anchors_dataframe = anchors_dataframe[anchors_dataframe.score > 0.0]
        anchors_dataframe["dists"] = 0.
        anchors_dataframe["cell1"] -= 1
        anchors_dataframe["cell2"] -= 1

        for i in anchors_dataframe.index:
            anchors_dataframe["dists"][i] = self.inter_dists[
                int(anchors_dataframe["cell1"][i]),
                int(anchors_dataframe["cell2"][i])]

        anchors_dataframe.sort_values("dists", inplace = True)

        self.anchors_ = anchors_dataframe.copy()
        self.query_ref_matchings = {}
        self.matched_ref = set()
        for i in np.unique(self.anchors_.cell1):
            self.query_ref_matchings[i] = int(self.anchors_[self.anchors_.cell1 == i].values[0][1])
            self.matched_ref.add(self.query_ref_matchings[i])


    def integrate_datasets(self):
        print("\tIntegration...")
        self.min_inter = np.min(self.inter_dists)
        self.max_inter = np.max(self.inter_dists)

        self.int_vectors = []
        self.alphas = {}
        self.alphas_na = {}

        self.q_neighbors_wa = {}
        self.q_neighbors_woa = {}

        self.q_neighbors_dists_wa = {}
        self.q_neighbors_dists_woa = {}

        self.q_a_dists = {}
        self.a_a_dists = {}

        self.k_neighbors = max(20, min(30, min(list(Counter(self.qkm.labels_).values())) - 1))


        for i in range(self.D_q.shape[0]):
            self.q_neighbors_wa[i], self.q_neighbors_dists_wa[i], self.a_a_dists[i], \
                self.q_a_dists[i], self.q_neighbors_woa[i], self.q_neighbors_dists_woa[
                i] = self.initializations("q_" + str(i))
        del self.Gq, self.Gr
        gc.collect()

        for i in range(self.D_q.shape[0]):
            self.build_int_vectors(i)

        self.prev_int_vectors = np.array(self.int_vectors)
        self.curr_int_vectors = np.zeros_like(self.prev_int_vectors)
        self.diff = [1000, np.sum(self.curr_int_vectors)]
        self.iteration = 0
        while abs(self.diff[-2] - self.diff[-1]) / (
                abs(self.diff[-1]) + 0.00001) > 0.00001 and self.iteration < self.max_iterations:
            self.iteration += 1
            for i in range(self.D_q.shape[0]):
                if i in self.query_ref_matchings:
                    self.curr_int_vectors[i] = self.int_vectors[i]
                    continue
                self.curr_int_vectors[i] = self.update_int_vectors(i, self.iteration)
            self.diff.append(np.sum(self.curr_int_vectors))
            self.prev_int_vectors = np.copy(self.curr_int_vectors)
        self.correct_query_dataset()

    def initializations(self, qi):

        def edge_weight(node):
            return node[1]

        qi_neighbors = set()
        for n_j in self.Gq.neighbors(qi):
            qi_neighbors.add((n_j, self.Gq.edges[qi, n_j]['dist']))
        qi_neighbors = sorted(qi_neighbors, key=edge_weight)[:self.k_neighbors]

        qi_neighbors_wa = []
        qi_neighbors_woa = []
        qi_neighbors_dists_wa = []
        qi_neighbors_dists_woa = []

        qi_a_distances = []
        ai_aj_neighbors = []

        if int(qi[2:]) in self.query_ref_matchings:
            anchor_of_qi = 'r_' + str(self.query_ref_matchings[int(qi[2:])])
            qi_neighbors_wa.append(int(qi[2:]))
            qi_neighbors_dists_wa.append(0)
            ai_aj_neighbors.append(0)
            qi_a_distances.append({self.query_ref_matchings[int(qi[2:])]: (self.inter_dists[int(qi[2:]),
            self.query_ref_matchings[int(qi[2:])]] - self.min_inter) / ( self.max_inter - self.min_inter)})

        else:
            anchor_of_qi = None

        for i in range(len(qi_neighbors)):
            qp = qi_neighbors[i][0]
            if int(qp[2:]) in self.query_ref_matchings:
                qi_neighbors_wa.append(int(qp[2:]))
                qi_neighbors_dists_wa.append(qi_neighbors[i][1])

                anchor_of_qp = 'r_' + str(self.query_ref_matchings[int(qp[2:])])
                if anchor_of_qi == anchor_of_qp:
                    ai_aj_neighbors.append(0)
                elif (anchor_of_qi, anchor_of_qp) in self.Gr.edges():
                    ai_aj_neighbors.append(self.Gr.edges[anchor_of_qi, anchor_of_qp]['dist'])
                else:
                    ai_aj_neighbors.append(1.)

                qi_a_distances.append({self.query_ref_matchings[int(qp[2:])]: (self.inter_dists[int(qp[2:]), self.query_ref_matchings[int(qp[2:])]] - self.min_inter) / ( self.max_inter - self.min_inter)})

            else:
                qi_neighbors_woa.append(int(qp[2:]))
                qi_neighbors_dists_woa.append(qi_neighbors[i][1])

        return qi_neighbors_wa, qi_neighbors_dists_wa, ai_aj_neighbors, qi_a_distances, qi_neighbors_woa, qi_neighbors_dists_woa


    def get_alphas(self, qi):
        dists = list((np.array(self.q_neighbors_dists_wa[qi]) + np.array(self.a_a_dists[qi])) / 2)
        dists_na = list(np.array(self.q_neighbors_dists_woa[qi]))

        ai_aj_dists = self.q_a_dists[qi]

        alphas = []
        alphas_na = []

        for i in range(len(dists)):
            v = self.beta * math.exp(-dists[i]) + (1 - self.beta) * math.exp(- list(ai_aj_dists[i].values())[0])
            alphas.append(v)

        for i in range(len(dists_na)):
            v = self.beta * math.exp(-dists_na[i])
            alphas_na.append(v)
        sum_alphas = np.sum(alphas) + np.sum(alphas_na)

        alphas = [[x / sum_alphas] for x in alphas]
        alphas_na = [[x / sum_alphas] for x in alphas_na]

        return alphas, alphas_na


    def build_int_vectors(self, qi):
        self.alphas[qi], self.alphas_na[qi] = self.get_alphas(qi)
        int_vector = np.zeros(self.D_q.shape[1])

        for v, w in zip(self.alphas[qi], self.q_neighbors_wa[qi]):
            int_vector += v[0] * (np.array(self.D_r.loc[self.D_r.index[self.query_ref_matchings[w]]]) -
                                  np.array(self.D_q.loc[self.D_q.index[w]]))
        self.int_vectors.append(int_vector)


    def update_int_vectors(self, qi, iteration):
        int_vector = np.zeros(self.D_q.shape[1])
        for v, w in zip(self.alphas[qi], self.q_neighbors_wa[qi]):
            int_vector += v[0] * (self.prev_int_vectors[w])

        if iteration != 1:
            for v, w in zip(self.alphas_na[qi], self.q_neighbors_woa[qi]):
                int_vector += v[0] * (self.prev_int_vectors[w])

        return int_vector


    def correct_query_dataset(self):
        self.D_q += np.array(self.curr_int_vectors)
