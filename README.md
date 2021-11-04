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

Also, the [help](clstr/README.md) options provide detailed descriptions:
```
docker run clustering_opt:latest main_clustering.py --help
```

---

## Reference
[Spalenza, M. A., Pirovani, J. P. C., and de Oliveira, E. (2019).  Structures Discovering for Optimizing External Clustering Validation Metrics. In Proceedings of the 19th International Conference on Intelligent Systems Designand Applications, volume 19 ofISDA 2019, pages 150–161, Auburn (WA),USA. Springer International Publishing.](https://link.springer.com/chapter/10.1007/978-3-030-49342-4_15)
