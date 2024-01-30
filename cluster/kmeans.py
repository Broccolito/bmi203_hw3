import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.error = None


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        if len(mat.shape) != 2:
            raise ValueError("Input matrix must be two-dimensional")
        
        num_samples, num_features = mat.shape
        # Randomly initialize centroids
        self.centroids = mat[np.random.choice(num_samples, self.k, replace=False)]

        for _ in range(self.max_iter):
            # Assign each sample to the nearest centroid
            distances = cdist(mat, self.centroids)
            closest_centroids = np.argmin(distances, axis=1)

            # Calculate new centroids and error
            new_centroids = np.array([mat[closest_centroids == k].mean(axis=0) for k in range(self.k)])
            self.error = np.mean(np.min(distances, axis=1))

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break

            self.centroids = new_centroids


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        if self.centroids is None:
            raise RuntimeError("Fit the model before prediction")

        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("The input matrix must have the same number of features as the fit data")

        distances = cdist(mat, self.centroids)
        return np.argmin(distances, axis=1)


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        if self.error is None:
            raise RuntimeError("Fit the model before getting the error")
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        if self.centroids is None:
            raise RuntimeError("Fit the model before getting the centroids")
        return self.centroids