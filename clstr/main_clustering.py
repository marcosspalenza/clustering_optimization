import os
import sys
import time
import datetime
import warnings
import argparse
import scipy as sp
import numpy as np
from cluster_information import Clustering
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.metrics import  pairwise_distances, normalized_mutual_info_score, adjusted_rand_score, accuracy_score


"""
GLOBALS
"""
# Clustering
_ALGORITHMS = [ "agglomerative"] # "spectral", "dbscan", "kmeans"


# Optimizer
_OPTIMIZERS = [ "gprocess", "dummy", "dtree", "exhaustive"]

# Internal Clustering Validation Measures
_IV_INDEXES = ["silhouette", "ch_score", "db_score", "sse", "cv_size", "cv_distance"]

# Distance Affinity Metrics
"""
Warnning:
- The following metrics are simetric to other traditional distances : "cityblock", "l1", "l2" ~ "manhattan", "euclidean", "sqeuclidean"
- The following metrics are weighted and not addapted on system recognition format : "yule", "wminkowski".


# _METRICS = [dist for dist in pairwise_distances.__globals__["_VALID_METRICS"]] # deprecated global
"""
_METRICS = [
    'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
    'braycurtis', 'canberra', 'chebyshev','correlation', 'dice', 'hamming',
    'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    'yule'
]

"""
External Clustering Validation Measures
"""
_EV_INDEXES =["ARI", "NMI", "CA"]
#"nmi", "ari", "max_pairwise","min_pairwise"


def load_labels(filename, pathto="./data/IN/", header=0):
    labels = []
    with open(pathto+filename, 'r') as fh:
        docs = fh.read().split('\n')
    while header > 0:
        docs = docs[1:]
        header = header - 1
    labels = [l for l in docs if l != ""]
    return labels


def load_matrix(filename, fformat, pathto="./data/IN/", fsep=" ", header=1):
    data = []
    if fformat == "dense":
        docs = []
        with open(pathto+filename, 'r') as fh:
            docs = fh.read().split('\n')    
        head = docs[ : header]
        docs = docs[header : ]
        try:
            data = np.array([[float(l) for l in line.split(fsep)] for line in docs if line != ""])
        except Exception as err:
            print("[Error] Data I/O problems. "+str(err))
    elif fformat == "sparse":
        with open(pathto+filename, 'r') as fh:
            docs = fh.read().split('\n')    
        head = docs[ : header]
        docs = docs[header : ]
        docid = head[-1]
        try:
            data = np.zeros((int(docid.split(fsep)[0]), int(docid.split(fsep)[1])))
            for n in docs:
                if n != "":
                    id1, id2, n = n.split(fsep)
                    data[int(id1)-1, int(id2)-1] = float(n)
        except Exception as err:
            print("[Error] Data I/O problems. "+str(err))
    else:
        print("Unexpected File Format.")
        return None
    return data


"""
Analyze cluster labels to parse cluster_id in max vote labels
"""
def cluster_accuracy(true_labels, clusters):
    labels = np.array([uid for uid, ulab in enumerate(np.unique(true_labels)) for l in true_labels if ulab == l])
    cluster_classes = np.zeros(len(labels))
    for c in np.unique(clusters):
        cltr_id = [sid for sid, cid in enumerate(clusters) if cid == c]
        cltr_labels = labels[cltr_id]
        cltr_class = np.bincount(cltr_labels).argmax()
        for sid in cltr_id:
            cluster_classes[sid] = cltr_class
    return accuracy_score(labels, cluster_classes)


def main():
    parser = argparse.ArgumentParser(usage="%(prog)s [options] <dataset>", description="Clustering optimization by distance metrics and internal validity indexes.")
    parser.add_argument("dataset", help="Dataset folder.")
    parser.add_argument("-i", "--pathin", type=str, dest="dbinput", default="/data/input/", help="Dataset path location.")
    parser.add_argument("-o", "--pathout", type=str, dest="dboutput", default="/data/output/", help="Dataset results output.")
    parser.add_argument("-l", "--labels", type=str, dest="dblabels", default="", help="Dataset single column labels. Default : None")
    parser.add_argument("-s", "--sheader", type=int, dest="size_header", default=1, help="Dataset files header size. Default : 1.")
    parser.add_argument("-f", "--format", type=str, dest="dbformat", default="dense", help="Dataset input format. [Matrix Market (sparse), Dense Matrix (dense)]. Default : 'dense'.")
    parser.add_argument("-d", "--distance", type=str, dest="distance_metric", default="euclidean", help="Distance metric for samples' affinity evaluation \n"+"\t-".join(_METRICS))
    # Distance matrix not supported in this version
    parser.add_argument("-a", "--algorithm", type=str, dest="algorithm", default="agglomerative", help="Clustering algorithm. Default ': agglomerative. '. \n"+"-".join(_ALGORITHMS))
    # Algorithms' choice not supported in this version
    parser.add_argument("-t", "--optimizer", type=str, dest="optimizer", default="gprocess", help="Optimizer algorithm. Default : 'gprocess'. \n"+"-".join(_OPTIMIZERS))
    parser.add_argument("-e", "--evaluation", type=str, dest="evaluation", default="silhouette", help="Optimization measure using Clustering Internal Validity Indexes. Default : 'silhouette'. \n"+"-".join(_IV_INDEXES))
    parser.add_argument("-k", "--kclusters", type=int, dest="k_clusters", default=0, help="Initial k value tested as the optimization starter")
    args = parser.parse_args()

    assert args.dbformat == "sparse" or args.dbformat == "dense", "Invalid matrix format"
    assert args.distance_metric in _METRICS, "A invalid distance metric was selected"
    assert args.algorithm in _ALGORITHMS, "This clustering algorithm currently not supported"
    assert args.optimizer in _OPTIMIZERS, "This optimizer currently not avaliable"
    assert args.evaluation in _IV_INDEXES, "Internal Validity Index not found."
    assert args.dbinput != "" and args.dboutput != "", "Insert a valid input and output folder."

    if args.distance_metric in ["yule", "wminkowski"]:
        print("[Warning] This metric was not suited for the modelling process and may not work.")
    if args.distance_metric in ["cityblock", "l1", "l2"]:
        print("[Warning] This metric is symmetric to other avaliable distance, ensure the test consistency.")

    # Check libraries' warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    timer = 0.
    metrics = []
    clabels = []
    try:
        data = []
        if os.stat(args.dbinput+args.dataset):
            print("[Process] Loading dataset.")
            data = load_matrix(args.dataset, args.dbformat, pathto=args.dbinput, header=args.size_header)

        labels = []
        if args.dblabels != "":
            if os.stat(args.dbinput+args.dblabels):
                print("[Process] Loading data labels.")
                labels = load_labels(args.dblabels, pathto=args.dbinput, header=args.size_header)

        if data == []:
            print("[Error] Load data fails.")
        else:
            # Default use within Clustering process
            # print("[Process] Generating distance matrix")
            # distance_mtx = pairwise_distances(data, metric=args.distance_metric, n_jobs=None)
            print("[Process] Dataset Loaded. Size : "+str(data.shape))
            print("[Process] Generating Clustering module.")
            start_time = time.time()
            clstr = None
            evi_nmi = -1.0
            evi_ari = -1.0
            evi_ca = -1.0
            ivi_ss = -1.0
            ivi_db = -1.0
            ivi_ch = -1.0
            ivi_sse = -1.0
            ivi_cvs = -1.0
            if not os.path.isfile(args.dboutput+"exec.csv"):
                output = (
                    "\t".join([
                        "Dataset", "Date", "Time(min)", "Distance", "Algorithm",
                        "Index", "Optimizer", "SS", "DBS", "CHS", "SSE", "CVS",
                        "NMI", "ARI", "CA", "Clusters",
                    ])
                    +"\n"
                )
                with open(args.dboutput+"exec.csv", "w") as run:
                    run.write(output)
            clstr = Clustering(
                data, args.dbinput, args.dboutput, n_clusters_ = args.k_clusters, metric_ = args.distance_metric,
                algorithm_ = args.algorithm, optimization_ = args.evaluation,
                method_ = args.optimizer, labels_=labels
            )
            print("[Process] Starting Clustering.")
            clabels = clstr.cluster_analysis()
            print("[Process] Evaluating Clusters.")
            timer = (time.time() - start_time) / 60
            ivi_ss = clstr._silhouette(np.array(clabels))
            ivi_ch = clstr._davies_bouldin(np.array(clabels))
            ivi_db = clstr._calinski_harabasz(np.array(clabels))
            ivi_sse = clstr._sse(np.array(clabels))
            ivi_cvs = clstr._cv_size(np.array(clabels))
            if len(labels) > 0 and len(labels) == len(clabels):
                evi_ari = adjusted_rand_score(labels, clabels)
                evi_nmi = normalized_mutual_info_score(labels, clabels)
                evi_ca = cluster_accuracy(labels, clabels)
            print("[Process] Writing results.")
            metrics = [ivi_ss, ivi_db, ivi_ch, ivi_sse, ivi_cvs, evi_nmi, evi_ari, evi_ca]
            output = " ".join([str(c) for c in clabels])+"\n"
            with open(args.dboutput+"clusters.txt", "a") as run:
                run.write(output)
    except Exception as err:
        print("[Error] "+str(err))
        output = (
            "\t".join([
                args.dataset,
                str(datetime.datetime.now()),
                str(timer),
                args.distance_metric,
                args.algorithm,
                args.evaluation,
                args.optimizer,
                str(err).replace("\n","\t")
            ])
            +"\n"
        )
        with open(args.dboutput+"exceptions.csv", "a") as run:
            run.write(output)
    else:
        cluster_distribution = ":".join(
            [str(len(np.where(clabels == l)[0])) for l in np.unique(clabels)]
        )
        cluster_metrics = "\t".join([str(round(m,4)) for m in metrics])
        output = (
            "\t".join([
                args.dataset,
                str(datetime.datetime.now()),
                str(timer),
                args.distance_metric,
                args.algorithm,
                args.evaluation,
                args.optimizer,
                cluster_metrics,
                cluster_distribution,
            ])
            +"\n"
        )
        print(output)
        with open(args.dboutput+"exec.csv", "a") as run:
            run.write(output)
    print("[Process] Done!")

if __name__=="__main__":
    main()
