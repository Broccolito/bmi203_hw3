import pytest
import numpy as np
from cluster import KMeans

"""
- Testing with k=0: The model should raise a ValueError.

- Testing with the number of observations less than k: 
The model should raise a ValueError or handle it gracefully.

- Testing with very high k: The model should be able to handle high values of k, 
though performance might be a consideration.

- Testing with high dimensionality: The model should handle datasets with a high number of features.

- Testing with a single dimension: The model should work correctly with datasets having only one feature.

- Additional scenarios: These might include testing with invalid input types or ensuring 
that methods like predict, get_error, and get_centroids are called only after fitting the model.
"""

def test_k_zero():
    with pytest.raises(ValueError):
        KMeans(k=0)
    pass

def test_observations_less_than_k():
    kmeans = KMeans(k=5)
    X = np.random.rand(3, 2)  # 3 observations, 2 features
    with pytest.raises(ValueError):  # or check for a warning
        kmeans.fit(X)
    pass

def test_high_k():
    kmeans = KMeans(k=1000)
    X = np.random.rand(100, 2)
    with pytest.raises(ValueError):
        kmeans.fit(X)  # Should run without errors
    pass

def test_high_dimensionality():
    kmeans = KMeans(k=3)
    X = np.random.rand(100, 1000)  # High dimensionality
    kmeans.fit(X)  # Should run without errors
    pass

def test_single_dimension():
    kmeans = KMeans(k=2)
    X = np.random.rand(100, 1)  # Single dimension
    kmeans.fit(X)  # Should run without errors
    pass

def test_invalid_input_type():
    kmeans = KMeans(k=3)
    X = "invalid input" # Test if the function will fail if the input does not have all the attributes
    with pytest.raises(AttributeError):
        kmeans.fit(X)
    pass

def test_methods_before_fit():
    kmeans = KMeans(k=3)
    X = np.random.rand(10, 2)
    with pytest.raises(RuntimeError):
        kmeans.predict(X)
    with pytest.raises(RuntimeError):
        kmeans.get_error()
    with pytest.raises(RuntimeError):
        kmeans.get_centroids()
    pass