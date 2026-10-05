"""
train_evaluate.py
------------------
Trains DT, RF, SVM, KNN on the (a) original features and (b) PCA-reduced
features (Fm components, from Algorithm 2), then reports Precision,
Recall, Accuracy and F1-score for each -- reproducing Tables III-VI and
Figures 3-6 of the paper.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

RANDOM_STATE = 42

MODELS = {
    "DT":  lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RF":  lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "SVM": lambda: SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}


def evaluate_models(X: np.ndarray, y: np.ndarray, feature_set_name: str,
                     test_size: float = 0.25, scale: bool = True):
    """Train each of DT/RF/SVM/KNN and return a metrics DataFrame."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    rows = []
    for name, build in MODELS.items():
        clf = build()
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_test)
        infer_time = time.time() - t0

        rows.append({
            "FeatureSet": feature_set_name,
            "Algorithm": name,
            "Precision(%)": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Recall(%)": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Accuracy(%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "F1(%)": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "TrainTime(s)": round(train_time, 4),
            "InferTime(s)": round(infer_time, 4),
        })
        print(f"[{feature_set_name}] {name}: "
              f"Prec={rows[-1]['Precision(%)']}%  Rec={rows[-1]['Recall(%)']}%  "
              f"Acc={rows[-1]['Accuracy(%)']}%  F1={rows[-1]['F1(%)']}%  "
              f"(train={train_time:.3f}s, infer={infer_time:.4f}s)")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from ml.dataset import load_dataset

    X, y, cols, source = load_dataset()
    results = evaluate_models(X.values, y.values, "Original")
    print(results)
