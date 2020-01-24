# Clustering Optimization
An optimization method for clustering hyperparameters selection using internal validity indexes.

## Requirements: 
Docker Container Ubuntu 18.04

- Python 3.6

- [Skopt](https://scikit-optimize.github.io/)

- [Sklearn](https://flask.palletsprojects.com/)

## Container

Create via Dockerfile:
```
docker build -t clustering_opt .
```

Clone via DockerHub:
```
docker push marcosspalenza/clustering_opt:lattest
```

## Usage
Requires input and output directories volumes as following.
```
docker run -v IN_DIR:/data/input -v OUT_DIR:/data/output/ clustering_opt:lattest main_clustering.py DATASET
```

Replace IN_DIR and OUT_DIR with your data path and DATASET with your database name.

To change data I/O and clustering test observe *help* options:
```
docker run clustering_opt:lattest main_clustering.py --help
```

### Running Options:
`main_clustering.py [options] <dataset>`
___

- Positional arguments:
> dataset : Dataset folder.
___

- Optional arguments:
> -h, --help, show this help message and exit.
>
> -i --pathin,
> Dataset path input.  The standard path inside a container is `/data/input/` mapped docker volume.
>
> -o --pathout,
> Dataset path output. The standard path inside a container is `/data/output/` mapped docker volume.
>
> -l --labels,
> Dataset classes in a single column file. Default : `None`. Required to evaluate External Validity Indexes.
>
> -s --sheader,
> Dataset files header size. Default : `1`. Indicates documents' header lines to be ignored. Labels and dataset files using different header sizes will be misinterpreted.
>
> -f --format,
> Dataset input format. Two numeric dataset formats are supported `sparse` [Matrix Market](https://math.nist.gov/MatrixMarket/) (line column value) non-zero features or `dense` containing all features of each samples' line.
>
> -d --distance, 
> Distance metric for samples' affinity evaluation. Default : `euclidean`. The [scikit distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) avaliable as presented in [Table 1](#table-metrics).
>
> -a --algorithm,
> Clustering algorithm. Default : `agglomerative`. The list containing supported algorithms are presented in [Table 2](#table-algorithms).
>
> -t --optimizer, Optimizer algorithm. Default : `gprocess`. The list containing the supported optimizers are presented in [Table 3](#table-optimizer).
>
> -e --evaluation, 
> Optimization measure using Clustering Internal Validity Indexes - IVI. Default : `silhouette`. All the validation measures included are presented in [Table 4](#table-ivi).

<h4 id="table-metrics">
 Table 1. Metrics
</h4> 

| Metrics       |                |               |               |               |                |
|---------------|:--------------:|:-------------:|:-------------:|:-------------:|:--------------:|
| braycurtis    | canberra       | chebyshev     | cityblock ¹   | correlation   | cosine         |
| dice          | euclidean      | hamming       | haversine     | jaccard       | kulsinski      |
| l1 ¹          | l2 ¹           | mahalanobis   | manhattan     | matching      | minkowski      |
| nan_euclidean | rogerstanimoto | russellrao    | seuclidean    | sokalmichener | sokalsneath    |
| sqeuclidean   | wminkowski ²   | yule          |               |               |                |

***¹ Some distances are simetric to others. Prevent to run duplicate experiments!***

***² The optimization may not work using some distance metrics due to the data sparsity, required weighting or unbalanced features.***

<h4 id="table-algorithms">
 Table 2. Algorithms
</h4>

| Algorithms    |
|---------------|
| agglomerative |

<h4 id="table-optimizers">
 Table 3. Optimizers
</h4>

| Optimizers    |                |                |
|---------------|:--------------:|:--------------:|
| gprocess      | rdn_forest     | dummy          |

<h4 id="table-ivi">
 Table 4. Internal Validity Indexes - IVI
</h4>

| IVI           |                |                |                |                |
|---------------|:--------------:|:--------------:|:--------------:|:--------------:|
| silhouette    | size_avg       | mean_pairwise  | ch_score       | db_score       | 
 
## Reference
M. A. Spalenza, J. P. C. Pirovani, E. Oliveira. *“Structures Discovering for Optimizing External Clustering Validation Metrics.”* 19th. International Conference on Intelligent Systems Design and Applications (ISDA). 2019.
