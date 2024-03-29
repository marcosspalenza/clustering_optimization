
# Clustering Options:
`main_clustering.py [options] <dataset>`

___

## Positional arguments:
dataset : Dataset matrix filename.

___

## Optional arguments:
-h --help, show this help message and exit.

-i --pathin,
Dataset path input.  The standard path inside a container is `/data/input/` mapped docker volume.

-o --pathout,
Dataset path output. The standard path inside a container is `/data/output/` mapped docker volume.

-l --labels,
Dataset classes in a single column file. Default : `None`. Required to evaluate External Validity Indexes.

-s --sheader,
Dataset files header size. Default : `1`. Indicates documents' header lines to be ignored. Labels and dataset files using different header sizes will be misinterpreted.

-f --format,
Dataset input format. Two numeric dataset formats are supported `sparse` [Matrix Market](https://math.nist.gov/MatrixMarket/) (line column value) non-zero features or `dense` containing all features of each samples' line.

-d --distance, 
Distance metric for samples' affinity evaluation. Default : `euclidean`. The [scikit distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) avaliable as presented in [Table 1](#table-metrics).

-a --algorithm,
Clustering algorithm. Default : `agglomerative`. The list containing supported algorithms are presented in [Table 2](#table-algorithms).

-t --optimizer, Optimizer algorithm. Default : `gprocess`. The list containing the supported optimizers are presented in [Table 3](#table-optimizer).

-e --evaluation, 
Optimization measure using Clustering Internal Validity Indexes - IVI. Default : `silhouette`. All the validation measures included are presented in [Table 4](#table-ivi).

-k --kclusters,
Initial test for clustering optimization. The starter k are adjusted to the optimization's sampling space. Using k = 0 the tests start using a random value. Default : `0`.

## Format

### Sparse
[Matrix market](https://docs.scipy.org/doc/scipy/reference/io.html#matrix-market-files) sparse matrix format. Header contains the matrix size (samples, features, total). The values are presented using line and column indexes for a value, delimited by a single space.
```
1000 10 5000
1 1 5
1 2 3
1 3 8
2 1 1
2 3 3
...
1000 3 10
```

### Dense
Dense matrix format for samples x features data. All samples are represented using ***floats*** or ***integers*** space-separated values. The standard contains a matrix size header.
```
1000 10
1 1 0 3 0 32 1 8 2 0
2 3 5 6 7 100 9 2 1 10
7 1 0 0 11 0 3 5 8 8
...
2 1 2 3 0 20 1 8 3 0
```

## Parameters Contents
<h4 id="table-metrics">
 Table 1. Metrics
</h4> 

| Metrics         |                |               |               |               |                |
|-----------------|:--------------:|:-------------:|:-------------:|:-------------:|:--------------:|
| braycurtis      | canberra       | chebyshev     | cityblock ¹   | correlation   | cosine         |
| dice            | euclidean      | hamming       | haversine     | jaccard       | kulsinski      |
| l1 ¹            | l2 ¹           | mahalanobis   | manhattan     | matching      | minkowski      |
| nan_euclidean ² | rogerstanimoto | russellrao    | seuclidean    | sokalmichener | sokalsneath    |
| sqeuclidean     | wminkowski ²   | yule ²        |               |               |                |

***¹ Some distances are symmetric to others. Prevent to run duplicated experiments!***

***² The optimization may not work using certain distance metrics due to the support of data distributions.***

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
| gprocess      | dtree          | dummy          |

<h4 id="table-ivi">
 Table 4. Internal Validity Indexes - IVI
</h4>

| IVI           |                |                |                |                    |                |
|---------------|:--------------:|:--------------:|:--------------:|:------------------:|:--------------:|
| silhouette    | sse            | ch_score       | db_score       | cv_distance ³      | cv_size ³      |

***³ cv - evaluate the coefficient of variation (CV) of the clustering instances. The CV functions are the ratio of standard deviation and the mean of the retrieved values on cluster sizes (cv_size) or pairwise distances (cv_distance). ***

- Silhouette Score

- Sum of Squared Errors

- Calinski-Harabazs Score

- Davies-Bouldin Score

- Coefficient of Variation : Intra-Cluster Distance

- Coefficient of Variation : Cluster Sizes


## References

[Rousseeuw, Peter J. *“Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis”*. Computational and Applied Mathematics 20: 53-65. 1987.](https://www.sciencedirect.com/science/article/pii/0377042787901257)

[Davies, David L.; Bouldin, Donald W. *“A Cluster Separation Measure”*. IEEE Transactions on Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227. 1979.](https://ieeexplore.ieee.org/document/4766909)

[Calinski, T.; Harabasz, J. *“A Dendrite Method for Cluster Analysis”*. Communications in Statistics. 1974.](https://www.tandfonline.com/doi/abs/10.1080/03610927408827101)