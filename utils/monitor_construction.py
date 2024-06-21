import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import KMeans
from .abstraction import *
from .runtime_monitor import *

def features_clustering_by_k_start(features, k):
    clustering_results = {}
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init="auto")
    kmeans.fit(features)
    clustering_results = kmeans.labels_
    return clustering_results

def monitor_construction_from_features(features, clustering_result):
    loc_boxes = []

    if len(features):
        n_dim = features.shape[1]
        # determine the labels of clusters
        num_clusters = np.amax(clustering_result) + 1
        clustering_labels = np.arange(num_clusters)
        
        # extract the indices of vectors in a cluster
        clusters_indices = []
        for k in clustering_labels:
            indices_cluster_k, = np.where(clustering_result == k)
            clusters_indices.append(indices_cluster_k)
        
        # creat local box for each cluster
        loc_boxes = [Box() for i in clustering_labels]
        for j in range(len(loc_boxes)):
            points_j = [(i, features[i]) for i in clusters_indices[j]]
            loc_boxes[j].build(n_dim, points_j)
    else:
        raise RuntimeError("There exists no feature for building monitor!!")

    # creat the monitor for class y at layer i
    monitor = Monitor(good_ref=loc_boxes)
    return monitor