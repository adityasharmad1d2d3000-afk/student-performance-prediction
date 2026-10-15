
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# Folders
DATA_PATH = Path("data/student-mat.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Put student-mat.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH, sep=";")

print("STUDENT PERFORMANCE PREDICTION")
print("=" * 40)
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum().sum())

# 2. Create target: 1 = pass, 0 = fail
# The dataset uses a final grade (G3) from 0 to 20.
df["pass"] = (df["G3"] >= 10).astype(int)

print("\nPass/fail counts:")
print(df["pass"].value_counts().rename(
    index={0: "Fail", 1: "Pass"}
))

# 3. EDA: target distribution
sns.countplot(data=df, x="pass")
plt.xticks([0, 1], ["Fail", "Pass"])
plt.title("Student Pass/Fail Distribution")
plt.xlabel("Outcome")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pass_fail_distribution.png", dpi=150)
plt.close()

# EDA: study time and final grade
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "studytime_vs_grade.png", dpi=150)
plt.close()

# 4. Separate inputs and target.
# G3 is excluded because it defines the pass/fail target.
X = df.drop(columns=["G3", "pass"])
y = df["pass"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 5. Preprocess features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Train logistic regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\nMODEL EVALUATION")
print("=" * 40)
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC-AUC: {auc:.3f}")
print("\nClassification report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Fail", "Pass"],
    zero_division=0
))

# 10. Save confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# 11. Save results
with open(OUTPUT_DIR / "results.txt", "w", encoding="utf-8") as file:
    file.write("Student Performance Prediction Results\n")
    file.write(f"Dataset rows: {len(df)}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write(f"ROC-AUC: {auc:.4f}\n\n")
    file.write(classification_report(
        y_test, y_pred,
        target_names=["Fail", "Pass"],
        zero_division=0
    ))

print("\nGraphs and results saved in the outputs folder.")
print("Project completed successfully!")
