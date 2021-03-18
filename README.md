# Clustering Optimization
An optimization method for clustering hyperparameters selection using internal validity indexes.

## Requirements: 
Docker Container Ubuntu 18.04

- Python 3

- [Skopt](https://scikit-optimize.github.io/)

- [Sklearn](https://scikit-learn.org/stable/index.html)

## Container

Create via Dockerfile:
```
docker build -t clustering_opt .
```

Clone via DockerHub:
```
docker pull marcosspalenza/clustering_opt
```

## Usage
Requires input and output directories volumes as following.

```
docker run -v IN_DIR:/data/input/ -v OUT_DIR:/data/output/ clustering_opt:latest main_clustering.py DATASET
```

Replace IN_DIR and OUT_DIR with your data path and DATASET with your database name.

For changes on data I/O and clustering test parameters observe [help](clstr/README.md) options:
```
docker run clustering_opt:latest main_clustering.py --help
```

---

## Reference
M. A. Spalenza, J. P. C. Pirovani, E. Oliveira. *“Structures Discovering for Optimizing External Clustering Validation Metrics.”* 19th. International Conference on Intelligent Systems Design and Applications (ISDA). 2019.
