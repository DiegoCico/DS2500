import csv
from Homework.Homework1.team import Team
import matplotlib.pyplot as plt


path = "./goals.csv"
all_teams = []

team_colors = {
    "Boston Fleet": "#002244",
    "New York Sirens": "#CC0033",
    "Montreal Victoire": "#003E7E",
    "Toronto Sceptres": "#00205B",
    "Minnesota Frost": "#6BAED6",
    "Ottawa Charge": "#FFCC00",
}


def part1_questions(all_teams):
    total_goals = {}
    for team in all_teams:
        total_goals[team.name] = team.get_total_goals()
    most_goals_team = None
    most_goals = float('-inf')

    for team_name, goals in total_goals.items():
        if goals > most_goals:
            most_goals = goals
            most_goals_team = team_name


    fewest_goals = min(total_goals.values())
    zero_point_games = 0
    for team in all_teams:
        for goal in team.goals:
            if goal == 0:
                zero_point_games += 1

    total_goals_sum = sum(total_goals.values())
    num_teams = len(all_teams)
    avg_goals = total_goals_sum / num_teams

    print(f"Team with most goals: {most_goals_team}")
    print(f"Fewest total goals by a team: {fewest_goals}")
    print(f"Games with zero points: {zero_point_games}")
    print(f"Average total goals: {avg_goals}")

def part2_visualization(all_teams):
    # 24 matches
    for i in range(24):
        plt.figure(figsize=(10, 6))
        plt.title(f"Game {i + 1}")
        plt.xlabel("Goals")

        for team in all_teams:
            team.move_next()
            team.draw()


        plt.legend(loc="upper left")
        plt.grid(False)
        plt.pause(2)
        plt.show(block=False)
        plt.figure()

if __name__ == "__main__":
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            team_name = row[0]
            goals = row[2:]
            team = Team(team_name, color=team_colors[team_name])

            for goal in goals:
                if goal.strip() == "" or goal.lower() == "nan":
                    team.add_goal(0)
                else:
                    team.add_goal(int(goal))

            all_teams.append(team)

    part1_questions(all_teams)
    part2_visualization(all_teams)