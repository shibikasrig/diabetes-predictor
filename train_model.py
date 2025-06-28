import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 2: Preprocess - Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train multiple models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")

    # Select best model
    if acc > best_score:
        best_score = acc
        best_model = model

# Step 5: Save the best model and scaler
if best_model:
    pickle.dump(best_model, open("diabetes_model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    print(f"\n✅ Model trained and saved with accuracy: {best_score:.2f}")
else:
    print("❌ No model was selected.")
