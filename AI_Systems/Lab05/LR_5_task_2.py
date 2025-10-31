import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(path: Path):
    data = np.loadtxt(path, delimiter=',', dtype=float)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y

def plot_decision_surface(ax, clf, X, y, title):
    pad = 0.7
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25)
    for lab, mk in [(0, 'o'), (1, 's')]:
        ax.scatter(X[y == lab, 0], X[y == lab, 1], s=45, marker=mk, label=f'class {lab}', edgecolor='k')
    ax.set_title(title); ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.legend()

def main():
    ap = argparse.ArgumentParser(description="LR_5_task_2 — дисбаланс класів (Завдання 2.2)")
    ap.add_argument("--data", type=str, default="data_imbalance.txt")
    ap.add_argument("--balance", type=str, choices=["off", "on"], default="off")
    ap.add_argument("--ignore", action="store_true", help="ігнорувати zero-division warnings")
    args = ap.parse_args()

    if args.ignore:
        warnings.filterwarnings("ignore")

    X, y = load_data(Path(args.data))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    kwargs = dict(n_estimators=100, max_depth=7, random_state=1)
    if args.balance == "on":
        clf = ExtraTreesClassifier(class_weight="balanced", **kwargs)
        title_suffix = " (balanced)"
    else:
        clf = ExtraTreesClassifier(**kwargs)
        title_suffix = " (unbalanced)"

    clf.fit(Xtr, ytr)

    fig, ax = plt.subplots(figsize=(6,5))
    plot_decision_surface(ax, clf, Xtr, ytr, f"Decision boundary{title_suffix}")
    plt.tight_layout(); plt.show()

    y_pred = clf.predict(Xte)
    print("Confusion matrix:\n", confusion_matrix(yte, y_pred))
    print("\nClassification report:\n", classification_report(yte, y_pred, digits=4))

if __name__ == "__main__":
    main()
