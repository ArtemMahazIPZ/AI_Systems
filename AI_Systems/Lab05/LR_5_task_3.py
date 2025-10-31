import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

DATA_FILE = "data_random_forests.txt"

def load_data(path: Path):
    data = np.loadtxt(path, delimiter=',', dtype=float)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y

def run_grid(X, y, scoring):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    clf = RandomForestClassifier(random_state=1)
    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 7]
    }
    gs = GridSearchCV(clf, param_grid=param_grid, scoring=scoring, cv=5, n_jobs=-1, return_train_score=True)
    gs.fit(Xtr, ytr)
    y_pred = gs.predict(Xte)

    print(f"\n=== Scoring: {scoring} ===")
    print("Best params:", gs.best_params_)
    print("Best CV score:", gs.best_score_)
    print("\nClassification report on test:\n", classification_report(yte, y_pred, digits=4))

    df = pd.DataFrame(gs.cv_results_)
    df = df[["params", "mean_test_score", "std_test_score", "mean_train_score"]]
    print("\nGrid summary (top 5 rows):")
    print(df.sort_values("mean_test_score", ascending=False).head())

def main():
    X, y = load_data(Path(DATA_FILE))
    run_grid(X, y, scoring="precision_macro")
    run_grid(X, y, scoring="recall_macro")

if __name__ == "__main__":
    main()
