"""
PRT565 Assessment 3 - Breast Cancer Diagnosis Prediction
Group SYD-G1-22


Compares five classifiers (Decision Tree, Random Forest, Gaussian Naive Bayes,
SVM with RBF kernel, and an MLP neural network) on the Breast Cancer Wisconsin
(Diagnostic) dataset.
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

OUT = os.environ.get("OUT_DIR", ".")

# ---------------------------------------------------------------
# 1. Load and inspect the data
# ---------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")      # 0 = malignant, 1 = benign

df = X.copy()
df["diagnosis"] = np.where(y == 0, "Malignant", "Benign")
df.to_csv(os.path.join(OUT, "breast_cancer_wisconsin.csv"), index=False)

print("Shape:", X.shape)
print("Missing values:", int(X.isnull().sum().sum()))
print("Duplicate rows:", int(X.duplicated().sum()))
print(y.map({0: "Malignant", 1: "Benign"}).value_counts())

# ---------------------------------------------------------------
# 2. Stratified 80/20 train-test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------------------------------------------
# 3. Feature scaling (fit on training data only)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Define the five models
# ---------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                               solver="adam", early_stopping=True,
                               max_iter=1000, random_state=42),
}
uses_unscaled = {"Naive Bayes"}

# ---------------------------------------------------------------
# 5. Train, predict and evaluate
# ---------------------------------------------------------------
rows, preds, probs = [], {}, {}
for name, model in models.items():
    Xtr, Xte = (X_train, X_test) if name in uses_unscaled else (X_train_s, X_test_s)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    preds[name], probs[name] = y_pred, y_prob
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })
    print(f"\n{name}\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred,
                                target_names=["Malignant", "Benign"], digits=4))

results = pd.DataFrame(rows).set_index("Model").round(4)
print(results)
results.to_csv(os.path.join(OUT, "model_results.csv"))

# ---------------------------------------------------------------
# 6. Feature importance from the Random Forest
# ---------------------------------------------------------------
importances = pd.Series(models["Random Forest"].feature_importances_,
                        index=X.columns).sort_values(ascending=False)
print("\nTop 10 features:\n", importances.head(10).round(4))
