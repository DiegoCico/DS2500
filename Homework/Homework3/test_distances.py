def test_euclidean():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0
    assert euclidean_distance([1, 2, 3], [4, 5, 6]) == math.sqrt(27)
    assert euclidean_distance([5, 5], [5, 5]) == 0.0

def test_manhattan():
    assert manhattan_distance([0, 0], [3, 4]) == 7
    assert manhattan_distance([1, 2, 3], [4, 5, 6]) == 9
    assert manhattan_distance([5, 5], [5, 5]) == 0

def test_jaccard():
    assert jaccard_index(set([1, 2, 3]), set([2, 3, 4])) == 2/4
    assert jaccard_index(set([1, 2]), set([1, 2, 3, 4])) == 2/4
    assert jaccard_index(set([1, 2]), set([3, 4])) == 0.0

def test_hamming():
    assert hamming_distance("1010", "1001") == 2
    assert hamming_distance("1100", "1100") == 0
    assert hamming_distance("1111", "0000") == 4