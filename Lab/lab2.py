import math

def euclidean(p, q):
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def manhattan(p, q):
    return sum(abs(pi - qi) for pi, qi in zip(p, q))

def hamming(p, q):
    return sum(pi != qi for pi, qi in zip(p, q))

def jaccard(A, B):
    intersection = sum(1 for ai, bi in zip(A, B) if ai == bi == 1)
    union = sum(1 for ai, bi in zip(A, B) if ai == 1 or bi == 1)
    return intersection / union if union != 0 else 0.0

def haversine(lat1, lon1, lat2, lon2, radius=6371000):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c
