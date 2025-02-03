import csv
import matplotlib.pyplot as plt
from distances import euclidean, manhattan, hamming, jaccard

def load_numeric_data(filename):
    """
    Load numeric data from a CSV file.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        dict: A dictionary mapping identifiers (str) to lists of floats.
    """
    data = {}
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data[row[0]] = list(map(float, row[1:]))
    return data

def load_binary_data(filename):
    """
    Load binary data from a CSV file.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        dict: A dictionary mapping identifiers (str) to lists of integers.
    """
    data = {}
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data[row[0]] = list(map(int, row[1:]))
    return data

def compute_pairwise_distances(data, distance_fn):
    """
    Compute pairwise distances for all unique pairs in the dataset.

    Args:
        data (dict): Dictionary mapping identifiers to data points.
        distance_fn (callable): Function to compute distance between two data points.

    Returns:
        list: A list of distances computed for each unique pair.
    """
    keys = list(data.keys())
    distances = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            distances.append(distance_fn(data[keys[i]], data[keys[j]]))
    return distances

def main():
    """
    Load datasets, compute distances, print the computed means, and generate scatter plots.
    """
    numeric_data = load_numeric_data("numeric.csv")
    binary_data = load_binary_data("binary.csv")

    row1 = input("Enter first row name: ").strip()
    row2 = input("Enter second row name: ").strip()
    if row1 not in numeric_data or row2 not in numeric_data:
        print("One or both row names not found in numeric data.")
    else:
        specific_distance = euclidean(numeric_data[row1], numeric_data[row2])
        print(f"Euclidean distance between {row1} and {row2}: {specific_distance:.4f}")

    if "Alpha" not in numeric_data:
        print("Alpha row not found in numeric data.")
    else:
        alpha = numeric_data["Alpha"]
        euclidean_vals = []
        manhattan_vals = []
        for key, values in numeric_data.items():
            if key != "Alpha":
                euclidean_vals.append(euclidean(alpha, values))
                manhattan_vals.append(manhattan(alpha, values))
        mean_euclidean = sum(euclidean_vals) / len(euclidean_vals)
        mean_manhattan = sum(manhattan_vals) / len(manhattan_vals)
        print(f"Mean Euclidean Distance from Alpha: {mean_euclidean:.4f}")
        print(f"Mean Manhattan Distance from Alpha: {mean_manhattan:.4f}")

        plt.figure(figsize=(10, 5))
        plt.scatter(euclidean_vals, manhattan_vals)
        plt.xlabel("Euclidean Distance from Alpha")
        plt.ylabel("Manhattan Distance from Alpha")
        plt.title("Euclidean vs. Manhattan Distance from Alpha")
        plt.grid(True)
        plt.savefig("plot1.png")
        plt.show()

    jaccard_vals = compute_pairwise_distances(binary_data, jaccard)
    hamming_vals = compute_pairwise_distances(binary_data, hamming)
    mean_jaccard = sum(jaccard_vals) / len(jaccard_vals)
    mean_hamming = sum(hamming_vals) / len(hamming_vals)
    print(f"Mean Jaccard Index: {mean_jaccard:.4f}")
    print(f"Mean Hamming Distance: {mean_hamming:.4f}")

    plt.figure(figsize=(10, 5))
    plt.scatter(jaccard_vals, hamming_vals)
    plt.xlabel("Jaccard Index")
    plt.ylabel("Hamming Distance")
    plt.title("Jaccard vs. Hamming Distance")
    plt.grid(True)
    plt.savefig("plot2.png")
    plt.show()

if __name__ == "__main__":
    main()
