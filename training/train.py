import os
import json
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# dataset
data = load_digits()
X, y = data.data, data.target

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# eval
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# save
os.makedirs("artifacts", exist_ok=True)

joblib.dump(model, "artifacts/model.pkl")

with open("artifacts/metrics.json", "w") as f:
    json.dump({
        "accuracy": acc,
        "n_features": X.shape[1],
        "n_classes": len(set(y))
    }, f)

print("Model and metrics saved in artifacts/")