import math
import pytest
from distances import euclidean, manhattan, hamming, jaccard

def test_euclidean():
    """
    Test the euclidean function with multiple cases.
    """
    assert math.isclose(euclidean([0, 0], [0, 0]), 0.0, rel_tol=1e-6)
    assert math.isclose(euclidean([0, 0], [3, 4]), 5.0, rel_tol=1e-6)
    expected = math.sqrt(27)
    assert math.isclose(euclidean([1, 2, 3], [4, 5, 6]), expected, rel_tol=1e-6)

def test_manhattan():
    """
    Test the manhattan function with multiple cases.
    """
    assert manhattan([0, 0], [0, 0]) == 0
    assert manhattan([0, 0], [3, 4]) == 7
    assert manhattan([1, 2, 3], [4, 5, 6]) == 9

def test_hamming():
    """
    Test the hamming function with multiple cases.
    """
    assert hamming([1, 0, 1], [1, 0, 1]) == 0
    assert hamming([1, 0, 1], [1, 1, 1]) == 1
    assert hamming([0, 0, 0], [1, 1, 1]) == 3

def test_jaccard():
    """
    Test the jaccard function with multiple cases.
    """
    assert math.isclose(jaccard([1, 0, 1, 0], [1, 0, 1, 0]), 1.0, rel_tol=1e-6)
    assert math.isclose(jaccard([1, 0, 1, 0], [0, 1, 0, 1]), 0.0, rel_tol=1e-6)
    expected = 1/3
    assert math.isclose(jaccard([1, 1, 0, 0], [1, 0, 1, 0]), expected, rel_tol=1e-6)
