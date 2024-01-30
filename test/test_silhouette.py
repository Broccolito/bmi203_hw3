import pytest
import numpy as np
from sklearn.metrics import silhouette_samples
from cluster import Silhouette

def test_silhouette_score_against_sklearn():
    # Generate sample data
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    labels = np.array([0, 0, 0, 1, 1, 1])

    # Calculate Silhouette scores using sklearn
    sklearn_scores = silhouette_samples(X, labels)

    # Calculate Silhouette scores using your implementation
    silhouette = Silhouette()
    custom_scores = silhouette.score(X, labels)

    # Assert that the results are similar
    print(custom_scores)
    print(sklearn_scores)
    np.testing.assert_allclose(custom_scores, sklearn_scores, rtol=1e-2)
    pass

def test_invalid_input_shapes():
    silhouette = Silhouette()
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)  # Invalid shape for y

    with pytest.raises(ValueError):
        silhouette.score(X, y)
    pass

def test_invalid_cluster_cases():
    silhouette = Silhouette()
    X = np.random.rand(10, 2)

    # All points in the same cluster
    y_same = np.zeros(10)
    with pytest.raises(ValueError):
        silhouette.score(X, y_same)

    # Each point in its own cluster
    y_unique = np.arange(10)
    with pytest.raises(ValueError):
        silhouette.score(X, y_unique)
    pass