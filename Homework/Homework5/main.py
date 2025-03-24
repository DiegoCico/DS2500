import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

def load_and_clean_data(election_path, demographics_path):
    election_df = pd.read_csv(election_path, sep='\t')
    demo_df = pd.read_csv(demographics_path)

    # Fix casing and spaces
    election_df['state'] = election_df['state'].str.strip().str.title()
    demo_df['State'] = demo_df['State'].str.strip().str.title()

    # Compute demographic features
    demo_df['pct_male'] = demo_df['TOT_MALE'] / demo_df['TOT_POP']
    demo_df['pct_female'] = demo_df['TOT_FEMALE'] / demo_df['TOT_POP']
    demo_df['pct_white'] = (demo_df['WA_MALE'] + demo_df['WA_FEMALE']) / demo_df['TOT_POP']
    demo_df['pct_black'] = demo_df['Black'] / demo_df['TOT_POP']
    demo_df['pct_hispanic'] = demo_df['Hispanic'] / demo_df['TOT_POP']

    # Keep only needed columns
    demo_df = demo_df[['State', 'pct_male', 'pct_female', 'pct_white', 'pct_black', 'pct_hispanic']]
    demo_df = demo_df.sort_values('State').reset_index(drop=True)

    return election_df, demo_df

def get_state_winners(election_df, year):
    year_df = election_df[election_df['year'] == year]
    winners = year_df.groupby(['state']).apply(lambda g: g[g['candidatevotes'] == g['candidatevotes'].max()]).reset_index(drop=True)
    winners = winners[['state', 'party_detailed']]
    winners = winners.sort_values('state').reset_index(drop=True)
    return winners

def prepare_model_data(election_df, demo_df, year):
    winners = get_state_winners(election_df, year)
    merged = pd.merge(demo_df, winners, left_on='State', right_on='state')
    X = merged[['pct_male', 'pct_female', 'pct_white', 'pct_black', 'pct_hispanic']]
    y = merged['party_detailed']
    return train_test_split(X, y, test_size=0.25, random_state=0), merged

def evaluate_k_values(X_train, X_test, y_train, y_test):
    best_k = 5
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
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    acc = accuracy_score(y_test, preds)
    f1_dem = report.get('Democratic', {}).get('f1-score', 0.0)
    return knn, preds, acc, f1_dem, report

def plot_confusion_matrix(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    plt.tight_layout()
    plt.savefig("confusion_matrix_heatmap.png")
    plt.show()

def plot_feature_scatter(X_train, X_test, y_train, y_test):
    plt.figure(figsize=(10,6))
    # Combine for plotting
    train = X_train.copy()
    train['label'] = y_train
    train['set'] = 'train'
    test = X_test.copy()
    test['label'] = y_test
    test['set'] = 'test'
    full = pd.concat([train, test])

    sns.scatterplot(data=full, x='pct_white', y='pct_black', hue='label', style='set')
    plt.title('Feature Scatterplot by Set and Label')
    plt.savefig("feature_scatter.png")
    plt.tight_layout()
    plt.show()

def main():
    election_path = '1976-2020-president.tab'
    demographics_path = 'demographics.csv'

    election_df, demo_df = load_and_clean_data(election_path, demographics_path)

    year = int(input("Enter a year to model (e.g. 2000): "))
    (X_train, X_test, y_train, y_test), merged = prepare_model_data(election_df, demo_df, year)

    print("Number of training states:", len(X_train))

    best_k, k_results = evaluate_k_values(X_train, X_test, y_train, y_test)
    print("Best k by accuracy:", best_k)

    knn, preds, acc, f1_dem, report = run_classifier(X_train, X_test, y_train, y_test, best_k)
    print("Accuracy:", acc)
    print("F1 Score (Democratic):", f1_dem)

    state = input("Enter a state from test set to check prediction: ").strip().title()
    state_demo = merged[merged['State'] == state][['pct_male', 'pct_female', 'pct_white', 'pct_black', 'pct_hispanic']]
    if not state_demo.empty:
        prediction = knn.predict(state_demo)[0]
        actual = merged[merged['State'] == state]['party_detailed'].values[0]
        print(f"Prediction for {state}: {prediction}")
        print(f"Actual result for {state}: {actual}")
    else:
        print("State not found or not in test set.")

    print("Max states predicted correctly:", max(k_results.values()) * len(y_test))
    plot_confusion_matrix(y_test, preds)
    plot_feature_scatter(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()