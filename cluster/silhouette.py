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

        # Compute the mean intra-cluster distance for each sample
        a = np.array([np.mean(cdist([X[i]], X[y == y[i]], 'euclidean')) for i in range(n)])

        # Compute the mean nearest-cluster distance for each sample
        b = np.array([np.min([np.mean(cdist([X[i]], X[y == label], 'euclidean')) for label in set(y) - {y[i]}]) for i in range(n)])

        # Silhouette score for each sample
        silhouette_scores = (b - a) / np.maximum(a, b)

        return silhouette_scores