
import time
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

INPUT_FILE = "income_data.txt"
RANDOM_STATE = 42
MAX_ROWS = None

def is_numeric(s: str) -> bool:
    try:
        float(s); return True
    except Exception:
        return False

t0 = time.time()
X, y = [], []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if MAX_ROWS and i >= MAX_ROWS:
            break
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 15 or "?" in parts:
            continue
        X.append(parts[:-1])
        y.append(parts[-1])

print(f"[INFO] Завантажено рядків: {len(X)} за {time.time()-t0:.2f}s")

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
X = X_enc.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)

t1 = time.time()
clf.fit(X_train, y_train)
print(f"[INFO] Навчання займало: {time.time()-t1:.2f}s")

y_pred = clf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision (weighted):", round(precision_score(y_test, y_pred, average="weighted"), 4))
print("Recall (weighted):", round(recall_score(y_test, y_pred, average="weighted"), 4))
print("F1 (weighted):", round(f1_score(y_test, y_pred, average="weighted"), 4))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
