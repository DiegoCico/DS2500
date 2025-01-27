import csv
import matplotlib.pyplot as plt

from Homework.Homework2.food import Food

path = "./food_recommendatiosn.csv"

def part1_questions(foods):
    print("give 2 names of restaurants press enter for each")
    food_dict = {food.name: food for food in foods}
    while True:
        restaurant1 = str(input("1. ")).lower()
        restaurant2 = str(input("2. ")).lower()

        if restaurant1 in food_dict or restaurant2 in food_dict:
            break
        else:
            print("not in the csv")

    distance = food_dict[restaurant1].distance(food_dict[restaurant2])

    ratings = 0
    for food in foods:
        ratings += float(food.rating)

    price = 0
    for food in foods:
        price += float(food.price)

    wait = 0
    for food in foods:
        if food.wait == 3:
            wait += 1

    laneys_bgood = Food("laney's b.good", "8am-7pm", "American", "vegan friendly", 5, 2, 4, 1)

    min_distance = float('inf')
    most_similar = None

    for food in foods:
        current_distance = laneys_bgood.distance(food)
        if current_distance < min_distance:
            min_distance = current_distance
            most_similar = food

    print(f"1. Euclidean distance: {distance}")
    print(f"2. mean ratings: {ratings/len(foods)}")
    print(f"3. mean price: {price/len(foods)}")
    print(f"4. wait: {wait}")
    print(f"5-6. The most similar Laney's B.Good is {most_similar.name} {most_similar} with a distance of {min_distance}")

def part2_visualization(foods):
    laneys_bgood = Food("Laney's B.Good", "8am-7pm", "American", "vegan friendly", 5, 2, 4, 1)
    plt.figure(figsize=(8, 6))
    ratings = [food.rating for food in foods]
    prices = [food.price for food in foods]

    plt.scatter(ratings, prices, label='Restaurants', color='blue', alpha=0.6)
    plt.scatter(laneys_bgood.rating, laneys_bgood.price, label="Laney's B.Good", color='red', s=100, edgecolors='black')

    plt.title("Rating vs Price Comparison")
    plt.xlabel("Rating")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plot1_rating_vs_price.png")
    plt.close()

    restaurant1 = foods[0]
    restaurant2 = foods[1]

    labels = ['Rating', 'Price', 'Wait', 'Proximity']
    values1 = [restaurant1.rating, restaurant1.price, restaurant1.wait, restaurant1.proximity]
    values2 = [restaurant2.rating, restaurant2.price, restaurant2.wait, restaurant2.proximity]

    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.plot(x, values1, label=restaurant1.name, marker='o')
    plt.plot(x, values2, label=restaurant2.name, marker='s')

    plt.title("Comparison of Two Restaurants")
    plt.xticks(x, labels)
    plt.ylabel("Values")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plot2_restaurant_comparison.png")
    plt.close()

if __name__ == "__main__":
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

    part1_questions(all_food)
    part2_visualization(all_food)
