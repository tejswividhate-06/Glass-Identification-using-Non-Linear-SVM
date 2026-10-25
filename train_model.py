
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 1: Load the dataset
columns = [
    "Id", "RI", "Na", "Mg", "Al",
    "Si", "K", "Ca", "Ba", "Fe", "Type"
]

df = pd.read_csv("dataset.csv", header=None, names=columns)

# Step 2: Remove the ID column
df = df.drop(columns=["Id"])

# Step 3: Separate features and target
X = df.drop(columns=["Type"])
y = df["Type"]

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Create and train the Non-Linear SVM
model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train_scaled, y_train)

# Step 7: Predict test results
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== NON-LINEAR SVM RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 9: Save the trained model and scaler
joblib.dump(model, "glass_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as glass_svm_model.pkl")
print("Scaler saved as scaler.pkl")

# Step 10: Plot feature importance-style visualization
# SVM does not provide direct feature importance here.
# This plot shows the distribution of two measured features.
plt.figure(figsize=(8, 5))

for glass_type in sorted(y.unique()):
    subset = df[df["Type"] == glass_type]
    plt.scatter(
        subset["RI"],
        subset["Al"],
        label=f"Glass Type {glass_type}",
        alpha=0.7
    )

plt.xlabel("Refractive Index (RI)")
plt.ylabel("Aluminium (Al)")
plt.title("Glass Identification Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("glass_dataset_plot.png")
plt.savefig("glass_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
