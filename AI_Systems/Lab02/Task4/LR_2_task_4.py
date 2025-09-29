import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

INPUT_FILE = "income_data.txt"
RANDOM_STATE = 42

def is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def load_income(path):
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != 15 or "?" in parts:
                continue
            X.append(parts[:-1])
            y.append(parts[-1])
    X = np.array(X, dtype=object)
    y = np.array(y)
    X_enc = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        if all(is_numeric(v) for v in col):
            X_enc[:, j] = col.astype(float)
        else:
            le = preprocessing.LabelEncoder()
            X_enc[:, j] = le.fit_transform(col)
    return X_enc.astype(float), y

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "acc": accuracy_score(y_test, pred),
        "prec_w": precision_score(y_test, pred, average="weighted", zero_division=0),
        "rec_w": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_w": f1_score(y_test, pred, average="weighted", zero_division=0),
    }

X, y = load_income(INPUT_FILE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=200, n_jobs=None),
    "LDA": LinearDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "CART": DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE),
    "NaiveBayes": GaussianNB(),
    "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    results[name] = evaluate(model, X_train, X_test, y_train, y_test)

# Друк у порядку спадання F1
print("\n=== Порівняння алгоритмів (sorted by F1-weighted) ===")
for name, metrics in sorted(results.items(), key=lambda kv: kv[1]["f1_w"], reverse=True):
    print(f"{name:18s}  acc={metrics['acc']:.4f}  prec_w={metrics['prec_w']:.4f}  rec_w={metrics['rec_w']:.4f}  f1_w={metrics['f1_w']:.4f}")
