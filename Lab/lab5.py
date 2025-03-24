import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

wine = load_wine()
X = wine.data
y = wine.target

print("Feature names:", wine.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

wine_89 = X[89].reshape(1, -1)
pred_89 = knn.predict(wine_89)[0]
actual_89 = y[89]
print(f"Predicted: {pred_89}, Actual: {actual_89}")
correct_89 = pred_89 == actual_89
print("Correct prediction for index 89:", correct_89)

y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

incorrect = sum(y_pred != y_test)
print(f"Incorrect predictions: {incorrect}")

X_small = X[:, :2]
X_train_small, X_test_small, y_train, y_test = train_test_split(X_small, y, random_state=0)

knn_small = KNeighborsClassifier(n_neighbors=6)
knn_small.fit(X_train_small, y_train)
y_pred_small = knn_small.predict(X_test_small)

df = pd.DataFrame(X_test_small, columns=[wine.feature_names[0], wine.feature_names[1]])
df["predicted"] = y_pred_small

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=wine.feature_names[0],
    y=wine.feature_names[1],
    hue="predicted",
    palette="Set1",
    data=df
)
plt.title("Predicted Classes on Test Set (First Two Features)")
plt.savefig("knn_scatterplot.png")
plt.show()
