import math

def euclidean(p, q):
    """
    Compute the Euclidean distance between two points.

    Args:
        p (iterable of float): Coordinates of the first point.
        q (iterable of float): Coordinates of the second point.

    Returns:
        float: The Euclidean distance.
    """
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def manhattan(p, q):
    """
    Compute the Manhattan distance between two points.

    Args:
        p (iterable of float): Coordinates of the first point.
        q (iterable of float): Coordinates of the second point.

    Returns:
        float: The Manhattan distance.
    """
    return sum(abs(pi - qi) for pi, qi in zip(p, q))

def hamming(p, q):
    """
    Compute the Hamming distance between two sequences.

    Args:
        p (iterable): The first sequence.
        q (iterable): The second sequence.

    Returns:
        int: The Hamming distance (number of positions at which the elements differ).
    """
    return sum(pi != qi for pi, qi in zip(p, q))

def jaccard(A, B):
    """
    Compute the Jaccard index between two binary sequences.

    Args:
        A (iterable of int): The first binary sequence (e.g., list of 0s and 1s).
        B (iterable of int): The second binary sequence.

    Returns:
        float: The Jaccard index, defined as the ratio of the count of positions where both A and B are 1
               to the count of positions where at least one is 1. Returns 0.0 if the union is zero.
    """
    intersection = sum(1 for ai, bi in zip(A, B) if ai == bi == 1)
    union = sum(1 for ai, bi in zip(A, B) if ai == 1 or bi == 1)
    return intersection / union if union != 0 else 0.0

def haversine(lat1, lon1, lat2, lon2, radius=6371000):
    """
    Compute the great-circle distance between two points on Earth using the haversine formula.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.
        radius (float, optional): Earth's radius in meters. Defaults to 6371000.

    Returns:
        float: The distance between the two points in meters.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c
