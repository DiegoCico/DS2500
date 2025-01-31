import math
import csv
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def manhattan_distance(point1, point2):
    return sum(abs(x - y) for x, y in zip(point1, point2))

def jaccard_index(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def hamming_distance(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def load_numeric_data(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header row
        data = {row[0]: list(map(float, row[1:])) for row in reader}
    return data

def load_binary_data(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip header row
        data = {row[0]: set(row[1:]) for row in reader}
    return data

def compute_pairwise_distances(data, distance_fn):
    keys = list(data.keys())
    distances = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):  # Avoid duplicates
            dist = distance_fn(data[keys[i]], data[keys[j]])
            distances.append(dist)
    return distances

def main():
    numeric_data = load_numeric_data("numeric.csv")
    binary_data = load_binary_data("binary.csv")

    # Part 1: Compute distances
    alpha_distances_euclidean = [euclidean_distance(numeric_data["Alpha"], numeric_data[key])
                                  for key in numeric_data if key != "Alpha"]
    alpha_distances_manhattan = [manhattan_distance(numeric_data["Alpha"], numeric_data[key])
                                  for key in numeric_data if key != "Alpha"]
    
    jaccard_distances = compute_pairwise_distances(binary_data, jaccard_index)
    hamming_distances = compute_pairwise_distances(binary_data, hamming_distance)

    # Compute means
    mean_euclidean = sum(alpha_distances_euclidean) / len(alpha_distances_euclidean)
    mean_manhattan = sum(alpha_distances_manhattan) / len(alpha_distances_manhattan)
    mean_jaccard = sum(jaccard_distances) / len(jaccard_distances)
    mean_hamming = sum(hamming_distances) / len(hamming_distances)

    # Print results
    print(f"Mean Euclidean Distance from Alpha: {mean_euclidean:.4f}")
    print(f"Mean Manhattan Distance from Alpha: {mean_manhattan:.4f}")
    print(f"Mean Jaccard Index: {mean_jaccard:.4f}")
    print(f"Mean Hamming Distance: {mean_hamming:.4f}")

    # Part 2: Visualization
    plt.figure(figsize=(10, 5))
    plt.scatter(alpha_distances_euclidean, alpha_distances_manhattan)
    plt.xlabel("Euclidean Distance from Alpha")
    plt.ylabel("Manhattan Distance from Alpha")
    plt.title("Euclidean vs. Manhattan Distance")
    plt.grid()
    plt.savefig("plot1.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(jaccard_distances, hamming_distances)
    plt.xlabel("Jaccard Index")
    plt.ylabel("Hamming Distance")
    plt.title("Jaccard vs. Hamming Distance")
    plt.grid()
    plt.savefig("plot2.png")
    plt.show()

if __name__ == "__main__":
    main()
