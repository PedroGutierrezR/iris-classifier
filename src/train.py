from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--random-state", type=int, default=42)

args = parser.parse_args()
random_state = args.random_state

# 1. load the dataset
iris = load_iris()

# 2. split it
print(iris.feature_names, iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=args.test_size, random_state=random_state)

# 3. train and predict
model = DecisionTreeClassifier(random_state=random_state)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

# 4. display results
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                 display_labels=iris.target_names)

display.plot()
path = Path(__file__).parent.joinpath("../outputs").resolve()
path.mkdir(parents=True, exist_ok=True)
plt.savefig(str(path) + "/confusion_matrix.png")

# 5. persist model
joblib.dump(model, str(path) + "/model.joblib")