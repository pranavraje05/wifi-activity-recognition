print("🔥 MODEL.PY IS RUNNING 🔥")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os


def load_ucihar():
    base_path = "data/UCI HAR Dataset"

    print("Checking dataset path:", base_path)
    print("Exists:", os.path.exists(base_path))

    X_train = pd.read_csv(f"{base_path}/train/X_train.txt", delim_whitespace=True, header=None)
    y_train = pd.read_csv(f"{base_path}/train/y_train.txt", header=None)

    X_test = pd.read_csv(f"{base_path}/test/X_test.txt", delim_whitespace=True, header=None)
    y_test = pd.read_csv(f"{base_path}/test/y_test.txt", header=None)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()


def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_ucihar()

    print("Training model...")

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    return model, acc, y_test, preds
