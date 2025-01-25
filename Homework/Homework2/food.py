import math

class Food:
    """
    Represents a food item with various attributes.
    """
    def __init__(self, name, hours, category, features, rating, price, wait, proximity):
        """
        Initialize a Food object.
        :param name: Name of the food.
        :param hours: Hours of operation.
        :param category: Category/type of food.
        :param features: Features of the food (string).
        :param rating: Rating of the food (int).
        :param price: Price level of the food (int).
        :param wait: Wait time for the food (int).
        :param proximity: Proximity to the user (int).
        """
        self.name = name.strip().lower()
        self.hours = hours
        self.category = category
        self.features = features
        self.rating = int(rating)
        self.price = int(price)
        self.wait = int(wait)
        self.proximity = int(proximity)

    def distance(self, other):
        """
        Calculate the Euclidean distance between two Food objects based on
        quantitative attributes.
        :param other: Another Food object to compare with.
        :return: Euclidean distance (float).
        """
        return math.sqrt(
            (self.rating - other.rating) ** 2 +
            (self.price - other.price) ** 2 +
            (self.wait - other.wait) ** 2 +
            (self.proximity - other.proximity) ** 2
        )

    def __str__(self):
        """
        String representation of the Food object.
        :return: A string with the name, hours, category, and features.
        """
        return f'{self.name} ({self.hours}, {self.category}, Features: {self.features})'
