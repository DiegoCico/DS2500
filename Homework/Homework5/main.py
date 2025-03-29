import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def get_and_clean_data(election_path, demographics_path):
    """
    Loads and prepares election and demographic data.

    Args:
        election_path (str): Path to election data file.
        demographics_path (str): Path to demographic data file.

    Returns:
        tuple: (election_df, cleaned demographic_df)
    """
    election_df = pd.read_csv(election_path, sep='\t')
    demo_df = pd.read_csv(demographics_path)
    # Renaming to make things easier lol
    demo_df.rename(columns={'STNAME': 'State'}, inplace=True)

    election_df['state'] = election_df['state'].str.strip().str.title()
    demo_df['State'] = demo_df['State'].str.strip().str.title()

    demo_df['pct_male'] = demo_df['TOT_MALE'] / demo_df['TOT_POP']
    demo_df['pct_female'] = demo_df['TOT_FEMALE'] / demo_df['TOT_POP']
    demo_df['pct_white'] = (demo_df['WA_MALE'] + demo_df['WA_FEMALE']) / demo_df['TOT_POP']
    demo_df['pct_black'] = demo_df['Black'] / demo_df['TOT_POP']
    demo_df['pct_hispanic'] = demo_df['Hispanic'] / demo_df['TOT_POP']

    demo_df = demo_df[['State', 'pct_male', 'pct_female', 'pct_white', 'pct_black', 'pct_hispanic']]
    demo_df = demo_df.sort_values('State').reset_index(drop=True)

    return election_df, demo_df

def get_state_winners(election_df, year):
    """
    Gets winning party per state for a given year.

    Args:
        election_df (DataFrame): Election data.
        year (int): Election year.

    Returns:
        DataFrame: State and winning party.
    """
    year_df = election_df[election_df['year'] == year]
    winners_list = []
    grouped = year_df.groupby('state')
    for state, group in grouped:
        max_votes = group['candidatevotes'].max()
        state_winner = group[group['candidatevotes'] == max_votes]
        winners_list.append(state_winner)

    winners = pd.concat(winners_list).reset_index(drop=True)
    winners = winners[['state', 'party_detailed']].sort_values('state').reset_index(drop=True)
    return winners

def prepare_model_data(election_df, demo_df, year):
    """
    Prepares features and labels for model training/testing.

    Args:
        election_df (DataFrame): Election data.
        demo_df (DataFrame): Demographic data.
        year (int): Election year.

    Returns:
        tuple: (train-test split data, merged DataFrame)
    """
    winners = get_state_winners(election_df, year)
    merged = pd.merge(demo_df, winners, left_on='State', right_on='state')
    X = merged[['pct_male', 'pct_female', 'pct_white', 'pct_black', 'pct_hispanic']]
    y = merged['party_detailed']
    return train_test_split(X, y, test_size=0.25, random_state=0), merged

def train_values(X_train, X_test, y_train, y_test):
    """
    Finds best k value for KNN based on accuracy.

    Args:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.

    Returns:
        tuple: (best k value, accuracy results per k)
    """
    best_k = None
    best_acc = 0
    results = {}

    for k in range(5, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[k] = acc
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k, results

def run_classifier(X_train, X_test, y_train, y_test, k):
    """
    Trains and evaluates a KNN classifier.

    Args:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
        k (int): Number of neighbors.

    Returns:
        tuple: (model, predictions, accuracy, f1_score_democratic, full report)
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    acc = accuracy_score(y_test, preds)
    f1_dem = report.get('DEMOCRAT', {}).get('f1-score', 0.0)
    return knn, preds, acc, f1_dem, report

def plot_heatmap(y_test, preds):
    """
    Plots confusion matrix heatmap.

    Args:
        y_test (Series): True labels.
        preds (array): Predicted labels.
    """
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Heatmap')
    plt.tight_layout()
    plt.savefig("confusion_matrix_heatmap.png")
    plt.show()

def plot_scatter(X_train, X_test, y_train, y_test):
    """
    Plots scatterplot of training and testing points.

    Args:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
    """
    plt.figure(figsize=(10, 6))
    train = X_train.copy()
    train['label'] = y_train
    train['set'] = 'train'
    test = X_test.copy()
    test['label'] = y_test
    test['set'] = 'test'
    full = pd.concat([train, test])

    sns.scatterplot(data=full, x='pct_white', y='pct_black', hue='label', style='set', s=100)
    plt.title('Scatterplot Set and Label')
    plt.tight_layout()
    plt.savefig("feature_scatter.png")
    plt.show()

def main():
    election_path = 'president.tab'
    demographics_path = 'demographics.csv'
    election_df, demo_df = get_and_clean_data(election_path, demographics_path)
    year = 1992
    print("RUNNING YEAR 1992")
    (X_train, X_test, y_train, y_test), merged = prepare_model_data(election_df, demo_df, year)
    print("Q1.1: Number of training states:", len(X_train))
    knn_k3, preds_k3, acc_k3, f1_dem_k3, report_k3 = run_classifier(X_train, X_test, y_train, y_test, 3)
    correct_k3 = int(round(acc_k3 * len(y_test)))
    print("Q1.2: When k=3, number of testing states predicted correctly:", correct_k3)
    print("Q1.3: When k=3, F1 Score for Democratic:", f1_dem_k3)
    best_k, k_results = train_values(X_train, X_test, y_train, y_test)
    print("Q1.5: Best k by accuracy (for k=5 to 10):", best_k)
    max_correct = int(max(acc * len(y_test) for acc in k_results.values()))
    print("Q1.6: Maximum number of states predicted correctly:", max_correct)
    knn_best, preds_best, acc_best, f1_dem_best, report_best = run_classifier(X_train, X_test, y_train, y_test, best_k)
    plot_heatmap(y_test, preds_best)
    plot_scatter(X_train, X_test, y_train, y_test)
    #STATE
    state = "Florida"
    if state:
        state_demo = merged[merged['State'] == state][['pct_male', 'pct_female', 'pct_white', 'pct_black', 'pct_hispanic']]
        if not state_demo.empty:
            prediction = knn_k3.predict(state_demo)[0]
            actual = merged[merged['State'] == state]['party_detailed'].values[0]
            print(f"Prediction for {state}: {prediction}")
            print(f"Actual result for {state}: {actual}")
        else:
            print("State not found or not in the test set.")
    else:
        print("State is empty")

if __name__ == '__main__':
    main()
