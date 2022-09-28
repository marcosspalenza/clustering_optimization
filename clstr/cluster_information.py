import sys
import os
import warnings
import collections
import scipy as sp
import numpy as np
from scipy import io
from skopt import forest_minimize, gp_minimize, dummy_minimize, BayesSearchCV
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, MiniBatchKMeans, AffinityPropagation, Birch, DBSCAN # KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances, normalized_mutual_info_score, adjusted_rand_score


"""
class Data:
    def __init__(self,data_, matrixpath_, outputpath_, metric_="cosine")
        self.data = data_
        self.metric = metric_
        self.matrixpath = matrixpath_
        self.outputpath = outputpath_
        self.distance_mtx = squareform(pdist(data_, metric_))
"""


class Clustering:

    def __init__(self, data_, matrixpath_, outputpath_=None, distance_=None, metric_="cosine", n_clusters_=0, optimization_="silhouette", algorithm_="spectral", method_="gprocess", labels_=[], idxfocus_=None):
        self.data = data_
        if distance_ == [] or distance_ == None:
            warnings.filterwarnings('ignore', message='Metric applies a coversion.')
            self.distance_mtx = pairwise_distances(data_, metric=metric_, n_jobs=None)
        else:
            self.distance_mtx = distance_
        self.idxfocus = idxfocus_
        self.metric = metric_
        self.method = method_
        self.matrixpath = matrixpath_
        if outputpath_ == None or outputpath_ == "":
            self.outputpath = matrixpath_
        else:
            self.outputpath = outputpath_
        self.optimization = optimization_
        self.labels = labels_
        self.algorithm = algorithm_
        self.cluster_min = 2
        self.cluster_max = self._choose_max_k(np.shape(data_)[0])
        if n_clusters_ < self.cluster_min:
            self.n_clusters = self.cluster_min
        elif n_clusters_ > self.cluster_max:
            self.n_clusters = self.cluster_max
        else:
            self.n_clusters = n_clusters_


    def _choose_max_k(self, dataset_size):
        '''
        [ Jiawei Han et.al. 2011 - Data Mining Concepts and Techniques]
        simple k_cluster guess method is [sqrt(dataset size/2)]
        'elbow method', metric evaluation to define better results achieved for each k value.
        '''
        k = int(round((dataset_size / 2) ** 0.5))
        if dataset_size < 5000:
            k = k * 2
        return k


    """
    Evaluation
    """
    def _sse(self, cluster_labels):
        # Initialise Sum of Squared Errors
        # add self.idxfocus method
        sse = 0
        if self.idxfocus != None:
            clusters = [cluster_labels[i] for i in self.idxfocus]
            for c in np.unique(clusters):
                cluster = np.where(clusters == c)[0]
                centroid = np.mean(self.data[cluster, :], axis=0).reshape(1, -1)
                sim = 0
                for c in cluster:
                    sim += pairwise_distances(self.data[c].reshape(1, -1), centroid, metric="l2", n_jobs=None)[0, 0] # l2 - squared euclidean
                sse = sse + (sim/len(cluster))
        else:
            for c in np.unique(cluster_labels):
                cluster = np.where(cluster_labels == c)[0]
                centroid = np.mean(self.data[cluster, :], axis=0).reshape(1, -1)
                sim = 0
                for c in cluster:
                    sim += pairwise_distances(self.data[c].reshape(1, -1), centroid, metric="l2", n_jobs=None)[0, 0] # l2 - squared euclidean
                sse = sse + (sim/len(cluster))
        return sse


    def _silhouette(self, cluster_labels):
        if self.idxfocus != None:
            return silhouette_score(self.distance_mtx[self.idxfocus, :][:, self.idxfocus], cluster_labels[self.idxfocus], metric="precomputed")
        else:
            return silhouette_score(self.distance_mtx, cluster_labels, metric="precomputed")


    def _davies_bouldin(self, cluster_labels):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.idxfocus != None:
                return davies_bouldin_score(self.data[self.idxfocus, :], cluster_labels[self.idxfocus])
            else:
                return davies_bouldin_score(self.data, cluster_labels)


    def _calinski_harabasz(self, cluster_labels):
        if self.idxfocus != None:
            return calinski_harabasz_score(self.data[self.idxfocus, :], cluster_labels[self.idxfocus])
        else:
            return calinski_harabasz_score(self.data, cluster_labels)

    
    def _cv_distance(self, cluster_labels):
        cluster_dist = []
        if self.idxfocus != None:
            clusters = [cluster_labels[i] for i in self.idxfocus]
            for c in np.unique(clusters):
                cluster = np.where(clusters == c)[0]
                sim = []
                for id1, c1 in enumerate(cluster):
                    for id2, c2 in enumerate(cluster):
                        if id2 > id1: # usar precomputed
                            sim.append(self.distance_mtx[self.idxfocus[c1], self.idxfocus[c2]])
                if sim != []:
                    cluster_dist.append(np.mean(sim))
                else:
                    cluster_dist.append(0.0)
        else:
            for c in np.unique(cluster_labels):
                cluster = np.where(cluster_labels == c)[0]
                sim = []
                for id1, c1 in enumerate(cluster):
                    for id2, c2 in enumerate(cluster):
                        if id2 > id1: # usar precomputed
                            sim.append(self.distance_mtx[c1, c2])
                if sim != []:
                    cluster_dist.append(np.mean(sim))
                else:
                    cluster_dist.append(0.0)
        if np.mean(cluster_dist) == 0:
            return 0
        return np.std(cluster_dist)/np.mean(cluster_dist)


    def _cv_size(self, cluster_labels):
        cluster_size = []
        if self.idxfocus != None:
            clusters = [cluster_labels[i] for i in self.idxfocus]
            for c in np.unique(clusters):
                cltr = np.where(clusters == c)[0]
                cluster_size.append(len(cltr))
        else:
            for c in np.unique(cluster_labels):
                cltr = np.where(cluster_labels == c)[0]
                cluster_size.append(len(cltr))
        if np.mean(cluster_size) == 0:
            return 0
        return ((np.max(cluster_size) - np.min(cluster_size)) / len(cluster_size)) / sum(cluster_size)


    """
    Algorithms
    """
    ## algorithms:  'cluto', 'libDocumento', 'agglomerative', 'spectral', 'kmeans', 'minibatch', 'birch', 'affinity', 'dbscan'

    def _run_agglomerative(self):
        return AgglomerativeClustering(n_clusters=self.n_clusters, linkage="complete", affinity="precomputed").fit_predict(self.distance_mtx)


    def _run_spectral(self):
        return SpectralClustering(n_clusters=self.n_clusters, eigen_solver=None, affinity="precomputed").fit_predict(self.distance_mtx)


    def _run_dbscan(self):
        """
        DBSCAN cannot integrate the module to solve all problems using its default configuration.
        The proximity distance method sometimes return as result all negative (outliers) or single cluster values.
        That is not expected on optimization module and number of clusters selection method.
        Is highly recommended to use DBSCAN just as baseline model while not supported.
        
        DBSCAN issues example:
        >>> [-1, -1, -1, ..., -1, -1, -1]
        >>> [0, 0, 0, ..., 0, 0, 0]
        In both cases we solve throwing a null vector as result [], this approach needs to be solved in an upper level (caller).
        """
        # float(1/self.n_clusters) may be evaluated by some tests to prove efficiency
        labels =  DBSCAN(eps=float(1/self.n_clusters), metric="precomputed").fit_predict(self.distance_mtx)
        return labels


    # def _run_cluto(self):
    #     '''
    #     CLUTO official page
    #     http://glaros.dtc.umn.edu/gkhome/cluto/cluto/overview
    #     '''
    #     '''
    #     CLUTO was called and the results was not as expected. In euclidean graph method, CLUTO returns a vector smaller than the dataset size.
    #     In this cases we return a null array to be discarded in the upper level (caller).
    #     '''
    #     if self.metric == "cosine":
    #         os.system("vcluster -sim='cos' -showfeatures "+self.matrixpath+"values.mtx "+str(self.n_clusters))
    #         return [int(t) for t in self.read_document("values.mtx.clustering."+str(self.n_clusters), self.matrixpath).split("\n") if t != ""]
    #     if self.metric == "euclidean":
    #         os.system("vcluster -sim='dist' -clmethod='graph' "+self.matrixpath+"values.mtx "+str(self.n_clusters))
    #     else:
    #         return []
    #     return [int(t) for t in self.read_document("values.mtx.clustering."+str(self.n_clusters), self.matrixpath).split("\n") if t != ""]

    # def _run_libDocumento(self, iterations=100):
    #     # libDocumento have to be in the same path
    #     code_path = os.path.realpath(__file__)[:-os.path.realpath(__file__)[::-1].find("/")]
    #     if self.metric == "cosine":
    #         os.system(code_path+"libDocumento_COS --clustering --algorithm kmeans --features "+self.matrixpath+"mmvalues.mtx -k "+str(self.n_clusters)+" --num-inter "+str(iterations))
    #     if self.metric == "euclidean":
    #         os.system(code_path+"libDocumento_ECL --clustering --algorithm kmeans --features "+self.matrixpath+"mmvalues.mtx -k "+str(self.n_clusters)+" --num-inter "+str(iterations))
    #     else:
    #         return []
    #     return [int(t) for t in self.read_document("output.clustering", self.matrixpath).split("\n") if t != ""]

    # def _run_kmeans(self):
    #     scores = []
    #     # all_distances, model_predictions, losses, is_initialized, init_op, training_op
    #     if self.metric == "cosine":
    #         with tf.Session() as sess:
    #             kmeans = tf.contrib.factorization.KMeans(tf.convert_to_tensor(self.data, dtype=tf.float32), self.n_clusters, distance_metric=tf.contrib.factorization.COSINE_DISTANCE)
    #             (all_scores, cluster_idx, clustering_scores, _, kmeans_init, kmeans_training_op) = kmeans.training_graph()
    #             init = tf.global_variables_initializer()
    #             sess.run(init)
    #             sess.run(kmeans_init)
    #             return sess.run(cluster_idx) [0]
    #     if self.metric == "euclidean":
    #         with tf.Session() as sess:
    #             kmeans = tf.contrib.factorization.KMeans(tf.convert_to_tensor(self.data, dtype=tf.float32), self.n_clusters, distance_metric=tf.contrib.factorization.SQUARED_EUCLIDEAN_DISTANCE)
    #             (all_scores, cluster_idx, clustering_scores, _, kmeans_init, kmeans_training_op) = kmeans.training_graph()
    #             init = tf.global_variables_initializer()
    #             sess.run(init)
    #             sess.run(kmeans_init)
    #             return sess.run(cluster_idx) [0]
    #     else:
    #         return []


    """
    Optimizer
    """
    def cluster_analysis(self):
        #'gaussian' not implemented yet
        # ['minibatch', 'birch', 'affinity'] just works on euclidean distances
        """
        ['cluto', 'libDocumento', 'agglomerative', 'spectral', 'kmeans', 'dbscan']:
        ['cluto', 'libDocumento', 'agglomerative', 'spectral', 'kmeans', 'dbscan', 'minibatch', 'birch', 'affinity', 'gaussian']:
                return []
        """
        #standard files, fixed names in clustering method call
        # if self.algorithm == "libDocumento":
        #     self.save_mm_matrix('mmvalues.mtx', self.data, self.matrixpath)
        # if self.algorithm == "cluto":
        #     self.save_matrix('values.mtx', self.data, self.matrixpath)
        # cluster_labels = []
        score, self.n_clusters, tests = self._optimize_n_clusters()
        # if self.algorithm == "cluto":
        #     cluster_labels = self._run_cluto()
        # elif self.algorithm == "libDocumento":
        #     cluster_labels = self._run_libDocumento()
        if self.algorithm == "agglomerative":
            cluster_labels = self._run_agglomerative()
        elif self.algorithm == "spectral":
            cluster_labels = self._run_spectral()
        elif self.algorithm == "kmeans":
            cluster_labels = self._run_kmeans()
        elif self.algorithm == "dbscan":
            cluster_labels = self._run_dbscan()
        else:
            return None, -1
        return cluster_labels, len(np.unique(tests))


    def _optimize_n_clusters(self):
        max_iter = self.cluster_max - self.cluster_min
        ntests = int(0.5 * (self.cluster_max - self.cluster_min))
        if self.method == "exhaustive" or max_iter < 30:
            # No-Optimize Full Test
            result = self._cluster_metric(self.cluster_min)
            best = self.cluster_min
            for k_val in range(self.cluster_min, self.cluster_max):
                run_result = self._cluster_metric([k_val])
                if run_result < result:
                    result = run_result
                    best = k_val
            return result, best, self.cluster_max - self.cluster_min
        elif self.method == "gprocess":
            # Gaussian Opt.
            # gp_minimize is a gaussian implementation similar to sklearn GridSearch
            res = gp_minimize(self._cluster_metric, [(self.cluster_min, self.cluster_max)], n_calls=ntests)
            # res.fun #score
            # res.func_vals #all tested scores
            return res.fun, res.x[0], res.x_iters
        elif self.method == "dtree":
            # Decision Tree Opt.
            res = forest_minimize(self._cluster_metric, [(self.cluster_min, self.cluster_max)], base_estimator='RF', n_calls=ntests)
            # res.fun #score
            # res.func_vals #all tested scores
            return res.fun, res.x[0], res.x_iters
        elif self.method == "dummy":
            # Random Opt.
            res = dummy_minimize(self._cluster_metric, [(self.cluster_min, self.cluster_max)], n_calls=ntests)
            return res.fun, res.x[0], res.x_iters


    def _cluster_metric(self, k_value):
        # Collect the return from skopt
        if type(k_value) == list:
            self.n_clusters = k_value[0]
        else:
            self.n_clusters = k_value
        warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
        # The clustering algorithm chosen in the object construction
        # if self.algorithm == "cluto":
        #     try:
        #         cluster_labels = self._run_cluto()
        #     except Exception as e:
        #         print("[ERROR] An error occoured while we tried to call CLUTO")
        # elif self.algorithm == "libDocumento":
        #     try:
        #         cluster_labels = self._run_libDocumento()
        #     except Exception as e:
        #         print("[ERROR] An error occoured while we tried to call libDocumento clustering")
        if self.algorithm == "agglomerative":
            cluster_labels = self._run_agglomerative()
        elif self.algorithm == "spectral":
            cluster_labels = self._run_spectral()
        elif self.algorithm == "dbscan":
            cluster_labels = self._run_dbscan()
        elif self.algorithm == "kmeans":
            cluster_labels = self._run_kmeans()
        # Use one of the following methods to optimize clustering parameters
        if self.optimization == "nmi":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return -1 * normalized_mutual_info_score(self.labels, cluster_labels)
        elif self.optimization == "ari":
            return -1 * adjusted_rand_score(self.labels, cluster_labels)
        elif self.optimization == "sse":
            return self._sse(cluster_labels)
        elif self.optimization == "cv_size":
            return self._cv_size(cluster_labels)
        elif self.optimization == "cv_distance":
            return self._cv_distance(cluster_labels)
        elif self.optimization == "ch_score":
            return self._calinski_harabasz(cluster_labels)
        elif self.optimization == "db_score":
            return self._davies_bouldin(cluster_labels)
        elif self.optimization =="silhouette":
            return -1 * self._silhouette(cluster_labels) # silhoutte score
        return None


    """
    IO Utils 
    """
    def read_document(self, filename, pathto):
        with open(pathto+filename,"r") as txt:
            return txt.read()


    def save_document(self, filename, filedata, pathto):
        with open(pathto+filename,"w", encoding=self.encoding) as txt:
            return txt.write("\n".join(filedata))


    def save_mm_matrix(self, filename, pathto):
        mm_values = []
        sum_ = 0
        for docid, line in enumerate(self.data):
            for enum, l in enumerate(line):
                if float(l) != 0.0:
                    mm_values.append(str(docid+1)+" "+str(enum+1)+" "+str(l))
                    sum_ += 1
        with open(pathto+filename, 'w') as arq:
            arq.write("% Matrix Market\n")
            arq.write(str(np.shape(self.data)[0])+" "+str(np.shape(self.data)[1])+" "+str(sum_)+"\n")
            for m in mm_values:
                arq.write(str(m)+"\n")


    def load_mm_matrix(self, filename, pathto):
        return io.mmread(pathto+filename)


    def output_check(self, folder):
        try:
            os.stat(folder)
        except:
            os.makedirs(folder)


    def save_csv_document(self, filename, pathto, delimiter_=';'):
        with open(pathto+filename, 'w', newline='\n') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=delimiter_, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for r in self.data:
                csvwriter.writerow(r)


    def read_matrix(self, filename, pathto, header=True):
        with open(pathto+filename, 'r', encoding=self.encoding) as fh:
            if header:
                var = fh.readline()
                docids, nfeatures = var.replace("\n","").split()
            doclist = []
            docs = fh.read()
            for did in range(int(docids)):
                doclist.append(np.array([float(n) for n in docs.split("\n")[did].split()]))
        return docids, nfeatures, doclist


    def save_matrix(self, filename, pathto, fmt="%.4g", delimiter=' '):
        with open(pathto+filename, 'w') as fh:
            fh.write(str(np.shape(self.data)[0])+" "+str(np.shape(self.data)[1])+"\n")
            for rid in range(np.shape(self.data)[0]):
                #lista = delimiter.join("0" if data[rid,idx] == 0 else fmt % data[rid,idx] for idx in  range(np.shape(data[rid])[1]))
                lista = delimiter.join("0" if self.data[rid, idx] == 0 else fmt % self.data[rid,idx] for idx in  range(np.shape(self.data[rid])[0]))
                fh.write(lista + '\n')
