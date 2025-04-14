import csv
import matplotlib.pyplot as plt
from Homework.Homework2.food import Food

path = "./food_recommendatiosn.csv"

def load_food_data(path):
    """
    Loads food data from a CSV file and returns a list of Food objects.
    Args:
        path (str): Path to the CSV file containing the food data.
    Returns:
        list: List of Food objects created from the CSV data.
    """
    all_food = []
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        all_names = []

        for row in reader:
            name = row[0].strip().lower()
            if name in all_names:
                continue
            all_names.append(name)
            rating = float(row[1])
            price = float(row[2])
            wait = int(row[3])
            proximity = int(row[4])
            hours = row[5]
            category = row[6]
            features = row[7]
            food = Food(name, hours, category, features, rating, price, wait, proximity)
            all_food.append(food)

    return all_food

def get_restaurant_input(food_dict):
    """
    Prompts the user for the names of two restaurants and ensures that the names exist in the provided food dictionary.
    Args:
        food_dict (dict): Dictionary mapping restaurant names to Food objects.
    Returns:
        tuple: Two restaurant names that exist in the dictionary.
    """
    while True:
        restaurant1 = input("1. ").lower()
        restaurant2 = input("2. ").lower()

        if restaurant1 in food_dict and restaurant2 in food_dict:
            return restaurant1, restaurant2
        else:
            print("Not in the CSV")

def calculate_statistics(foods):
    """
    Calculates statistics for ratings, prices, and wait times across all foods.
    Args:
        foods (list): List of Food objects.
    Returns:
        tuple: Mean ratings, prices, and the count of wait times.
    """
    ratings = 0
    prices = 0
    wait = 0

    for food in foods:
        ratings += food.rating
        prices += food.price
        if food.wait == 3:
            wait += 1

    mean_ratings = ratings / len(foods)
    mean_price = prices / len(foods)

    return mean_ratings, mean_price, wait

def find_most_similar_food(foods, target_food):
    """
    Finds the food item most similar to a given target food.
    Args:
        foods (list): List of Food objects.
        target_food (Food): The target Food object to compare others against.
    Returns:
        Food: The most similar Food object.
    """
    min_distance = float('inf')
    most_similar = None
    for food in foods:
        current_distance = target_food.distance(food)
        if current_distance < min_distance:
            min_distance = current_distance
            most_similar = food
    return most_similar, min_distance

def count_restaurants_with_wait_score(foods, min_wait_score=3):
    """
    Counts how many restaurants in the list have a wait score of at least the specified minimum.
    Args:
        foods (list): List of Food objects.
        min_wait_score (int): Minimum wait score threshold.
    Returns:
        int: The number of restaurants with a wait score >= min_wait_score.
    """
    count = 0
    for food in foods:
        if food.wait >= min_wait_score:
            count += 1
    return count

def part1_questions(foods):
    """
    Asks the user for two restaurants, calculates statistics, and prints the results.
    Args:
        foods (list): List of Food objects.
    """
    print("Give 2 names of restaurants (press Enter after each)")
    food_dict = {food.name: food for food in foods}
    restaurant1, restaurant2 = get_restaurant_input(food_dict)

    distance = food_dict[restaurant1].distance(food_dict[restaurant2])

    mean_ratings, mean_price, wait = calculate_statistics(foods)

    laneys_bgood = Food("Laney's B.Good", "8am-7pm", "American", "vegan friendly", 5, 2, 4, 1)
    most_similar, min_distance = find_most_similar_food(foods, laneys_bgood)

    print(f"1. Euclidean distance: {distance}")
    print(f"2. Mean ratings: {mean_ratings}")
    print(f"3. Mean price: {mean_price}")
    print(f"4. Wait: {wait}")
    print(f"5-6. The most similar to Laney's B.Good is {most_similar.name} with a distance of {min_distance}")

    count = count_restaurants_with_wait_score(foods)
    print(f"7. Number of restaurants with a wait score of at least 3: {count}")

def create_scatter_plot(foods, laneys_bgood):
    """
    Creates and saves a scatter plot comparing restaurant ratings and prices.
    Args:
        foods (list): List of Food objects.
        laneys_bgood (Food): The reference food object for comparison.
    """
    ratings = []
    prices = []

    for food in foods:
        ratings.append(food.rating)
        prices.append(food.price)

    plt.figure(figsize=(8, 6))
    plt.scatter(ratings, prices, label='Restaurants', color='blue', alpha=0.6)
    plt.scatter(laneys_bgood.rating, laneys_bgood.price, label="Laney's B.Good", color='red', s=100, edgecolors='black')

    plt.title("Rating vs Price Comparison")
    plt.xlabel("Rating")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plot1_rating_vs_price.png")
    plt.close()

def create_barchart(restaurant1, restaurant2):
    """
    Creates and saves a bar chart comparing two restaurants based on their ratings, prices, wait times, and proximity.
    Args:
        restaurant1 (Food): The first restaurant to compare.
        restaurant2 (Food): The second restaurant to compare.
    """
    labels = ['Rating', 'Price', 'Wait', 'Proximity']

    values1 = [restaurant1.rating, restaurant1.price, restaurant1.wait, restaurant1.proximity]
    values2 = [restaurant2.rating, restaurant2.price, restaurant2.wait, restaurant2.proximity]

    x = range(len(labels))

    x2 = []
    for i in x:
        x2.append(i + 0.35)

    plt.figure(figsize=(10, 6))
    width = 0.35
    plt.bar(x, values1, width, label=restaurant1.name, color='blue', alpha=0.7)
    plt.bar(x2, values2, width, label=restaurant2.name, color='green', alpha=0.7)

    plt.title("Comparison of Two Restaurants")
    plt.xticks(x, labels)
    plt.ylabel("Values")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plot2_restaurant_comparison_barchart.png")
    plt.close()

def part2_visualization(foods):
    """
    Generates and saves visualizations comparing restaurants.
    Args:
        foods (list): List of Food objects.
    """
    laneys_bgood = Food("Laney's B.Good", "8am-7pm", "American", "vegan friendly", 5, 2, 4, 1)

    create_scatter_plot(foods, laneys_bgood)

    restaurant1 = foods[0]
    restaurant2 = foods[1]

    create_barchart(restaurant1, restaurant2)

def main():
    """
    Main function to drive the program.
    Loads food data, processes part 1 questions, and generates visualizations for part 2.
    """
    all_food = load_food_data(path)

    part1_questions(all_food)
    part2_visualization(all_food)

if __name__ == "__main__":
    main()
