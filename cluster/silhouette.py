import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        if len(X.shape) != 2:
            raise ValueError("Input data X must be a two-dimensional array")
        if len(y.shape) != 1:
            raise ValueError("Input labels y must be a one-dimensional array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n = X.shape[0]
        K = len(np.unique(y))
        if K == 1 or K == n:
            # Silhouette score is not meaningful for one cluster or if each data point is its own cluster
            raise ValueError("Cannot compute silhouette score with only one cluster or if each point is its own cluster")

        # Euclidean distance between two points
        def euclidean_distance(point1, point2):
            return np.sqrt(np.sum((point1 - point2) ** 2))

        # Mean distance to all points in the same cluster
        def mean_intra_cluster_distance(sample, cluster):
            if len(cluster) <= 1:
                return 0
            return sum(euclidean_distance(sample, point) for point in cluster) / (len(cluster) - 1)

        # Mean distance to all points in the nearest cluster
        def mean_nearest_cluster_distance(sample, all_clusters, own_cluster):
            distances = []
            for cluster in all_clusters:
                if cluster is not own_cluster:
                    mean_distance = sum(euclidean_distance(sample, point) for point in cluster) / len(cluster)
                    distances.append(mean_distance)
            return min(distances)
        
        # Organize samples by cluster
        clusters = {}
        for sample, label in zip(X, y):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sample)

        # Calculate silhouette score for each sample
        silhouette_scores = []
        for sample, label in zip(X, y):
            own_cluster = clusters[label]
            a = mean_intra_cluster_distance(sample, own_cluster)
            b = mean_nearest_cluster_distance(sample, clusters.values(), own_cluster)
            score = (b - a) / max(a, b) if max(a, b) != 0 else 0
            silhouette_scores.append(score)

        # Return the mean silhouette score
        return silhouette_scores
