import os
import json
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# dataset
data = load_iris()
X, y = data.data, data.target

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# eval
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# Créer le dossier artifacts/
os.makedirs("artifacts", exist_ok=True)

# Sauvegarder le modèle dans artifacts/
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Générer metrics.json
with open("artifacts/metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

print("Model and metrics saved in artifacts/")
