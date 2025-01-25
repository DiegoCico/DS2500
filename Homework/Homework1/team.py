import matplotlib.pyplot as plt

class Team:

    def __init__(self, name, x=0, y=0, color="purple"):
        """
        Initialize a new Team instance.

        :param name: str, the name of the team (required).
        :param x: int, the x-position of the team on the matplotlib grid (default: 0).
        :param y: int, the y-position of the team on the matplotlib grid (default: 0).
        :param color: str, the team's color when rendered on matplotlib (default: "purple").
        """
        self.name = name
        self.goals = []
        self.x = x
        self.y = y
        self.color = color

    def draw(self):
        """
        Render the team on the matplotlib grid at its current x, y position and with its color.
        """
        plt.scatter(self.x, self.y, color=self.color, marker='X', s=100, label=self.name.split()[0])
        plt.text(self.x, self.y, self.name, fontsize=9, ha='left', va='center')

    def add_goal(self, goal):
        """
        Add a list of goals scored in games to the team's record.
        """
        self.goals.append(goal)

    def get_total_goals(self):
        """
        Return a integer of the total number of goals in the team's record'.

        :return: int, total number of goals.
        """
        return sum(self.goals)

    def move_next(self):
        """
        Move the team to the right by the amount found in the next position in its goals list.
        """
        if self.goals:
            next_move = self.goals.pop(0)
            self.x += next_move
        else:
            print("No more goals left to move.")

    def __str__(self):
        """
        Return a string representation of the team.

        :return: str, description of the team.
        """
        return f"Team: {self.name}, Goals: {self.get_total_goals()}, Position: ({self.x}, {self.y}), Color: {self.color}"
