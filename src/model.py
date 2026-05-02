import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_evaluate():
    data = pd.read_csv("data/sample_data.csv")

    X = data[["feature1", "feature2", "feature3"]]
    y = data["activity"]

    # simple split
    split = int(0.8 * len(data))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=["walking","sitting","falling"])

    return model, acc, cm, preds
