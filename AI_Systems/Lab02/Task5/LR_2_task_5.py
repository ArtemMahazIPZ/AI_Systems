import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

RANDOM_STATE = 0

iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

clf = RidgeClassifier(tol=1e-2, solver="sag", random_state=RANDOM_STATE)
clf.fit(Xtrain, ytrain)

ypred = clf.predict(Xtest)

print("Accuracy:", np.round(accuracy_score(ytest, ypred), 4))
print("Precision (weighted):", np.round(precision_score(ytest, ypred, average="weighted"), 4))
print("Recall (weighted):", np.round(recall_score(ytest, ypred, average="weighted"), 4))
print("F1 Score (weighted):", np.round(f1_score(ytest, ypred, average="weighted"), 4))
print("Cohen Kappa:", np.round(cohen_kappa_score(ytest, ypred), 4))
print("Matthews Corrcoef:", np.round(matthews_corrcoef(ytest, ypred), 4))
print("\nClassification Report:\n", classification_report(ytest, ypred, target_names=target_names))

cm = confusion_matrix(ytest, ypred)
plt.figure(figsize=(5.2, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("RidgeClassifier — Confusion Matrix (Iris)")
plt.tight_layout()
plt.show()
