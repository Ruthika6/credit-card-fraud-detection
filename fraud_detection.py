# ============================================================
#   CREDIT CARD FRAUD DETECTION
#   Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from collections import Counter


# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv("data/creditcard.csv")

print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nNull Values:\n{df.isnull().sum().sum()} total nulls")


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# Class distribution
fraud_count = df["Class"].value_counts()
print(f"\nClass Distribution:\n{fraud_count}")
print(f"\nFraud Percentage: {fraud_count[1] / len(df) * 100:.4f}%")

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(["Legitimate (0)", "Fraud (1)"], fraud_count.values,
            color=["steelblue", "crimson"], edgecolor="black")
axes[0].set_title("Class Distribution (Imbalanced)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(fraud_count.values):
    axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

# Transaction amount distribution
axes[1].hist(df[df["Class"] == 0]["Amount"], bins=50, alpha=0.6,
             color="steelblue", label="Legitimate", density=True)
axes[1].hist(df[df["Class"] == 1]["Amount"], bins=50, alpha=0.6,
             color="crimson", label="Fraud", density=True)
axes[1].set_title("Transaction Amount Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Amount")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].set_xlim([0, 2000])

plt.suptitle("Credit Card Fraud - EDA", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA plots saved as 'eda_plots.png'")

# Statistical summary
print(f"\nAmount Statistics for Legitimate Transactions:")
print(df[df["Class"] == 0]["Amount"].describe())
print(f"\nAmount Statistics for Fraudulent Transactions:")
print(df[df["Class"] == 1]["Amount"].describe())

# Correlation heatmap (top features)
plt.figure(figsize=(12, 8))
corr = df.corr()["Class"].drop("Class").sort_values(key=abs, ascending=False).head(15)
sns.barplot(x=corr.values, y=corr.index, palette="coolwarm")
plt.title("Top 15 Features Correlated with Fraud", fontsize=14, fontweight="bold")
plt.xlabel("Correlation")
plt.tight_layout()
plt.savefig("feature_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Correlation plot saved as 'feature_correlation.png'")


# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

# Scale 'Amount' and 'Time' columns
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])

# Drop original Amount and Time
df.drop(columns=["Amount", "Time"], inplace=True)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print(f"Features shape: {X.shape}")
print(f"Target distribution: {Counter(y)}")


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Train-Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train fraud count: {Counter(y_train)}")
print(f"Test fraud count : {Counter(y_test)}")


# ============================================================
# 5. HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Applying SMOTE (Oversampling)")
print("=" * 60)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {Counter(y_train)}")
print(f"After SMOTE : {Counter(y_train_res)}")


# ============================================================
# 6. MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training Multiple Models")
print("=" * 60)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                              random_state=42, n_jobs=-1),
}

results = {}

for name, model in models.items():
    print(f"\n--- Training: {name} ---")

    # Train
    model.fit(X_train_res, y_train_res)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
    }

    print(f"ROC-AUC Score    : {roc_auc:.4f}")
    print(f"Avg Precision    : {avg_prec:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")


# ============================================================
# 7. COMPARE MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Model Comparison")
print("=" * 60)

comparison_df = pd.DataFrame({
    name: {
        "ROC-AUC": v["roc_auc"],
        "Avg Precision": v["avg_precision"],
    }
    for name, v in results.items()
}).T

print(comparison_df.sort_values("ROC-AUC", ascending=False))

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(comparison_df))
width = 0.35
bars1 = ax.bar(x - width/2, comparison_df["ROC-AUC"], width, label="ROC-AUC", color="steelblue")
bars2 = ax.bar(x + width/2, comparison_df["Avg Precision"], width, label="Avg Precision", color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(comparison_df.index, rotation=15, ha="right")
ax.set_ylim([0.8, 1.01])
ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Model comparison saved as 'model_comparison.png'")


# ============================================================
# 8. BEST MODEL — DETAILED ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Best Model — Detailed Analysis")
print("=" * 60)

best_name = comparison_df["ROC-AUC"].idxmax()
best = results[best_name]
print(f"Best Model: {best_name} (ROC-AUC = {best['roc_auc']:.4f})")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"], ax=axes[0])
axes[0].set_title(f"Confusion Matrix\n{best_name}", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best["y_prob"])
axes[1].plot(fpr, tpr, color="darkorange", lw=2,
             label=f"AUC = {best['roc_auc']:.4f}")
axes[1].plot([0, 1], [0, 1], color="navy", linestyle="--")
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title(f"ROC Curve\n{best_name}", fontweight="bold")
axes[1].legend(loc="lower right")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best["y_prob"])
axes[2].plot(recall, precision, color="green", lw=2,
             label=f"AP = {best['avg_precision']:.4f}")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].set_title(f"Precision-Recall Curve\n{best_name}", fontweight="bold")
axes[2].legend(loc="upper right")

plt.suptitle(f"Best Model: {best_name}", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("best_model_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Best model analysis saved as 'best_model_analysis.png'")


# ============================================================
# 9. FEATURE IMPORTANCE (for tree-based best model)
# ============================================================
if hasattr(best["model"], "feature_importances_"):
    print("\n" + "=" * 60)
    print("STEP 9: Feature Importance")
    print("=" * 60)

    importances = pd.Series(
        best["model"].feature_importances_, index=X.columns
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title(f"Top 15 Feature Importances — {best_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Feature importance saved as 'feature_importance.png'")

    print(f"\nTop 10 Important Features:\n{importances.head(10)}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for name, v in sorted(results.items(), key=lambda x: -x[1]["roc_auc"]):
    print(f"{name:<25} | ROC-AUC: {v['roc_auc']:.4f} | Avg Precision: {v['avg_precision']:.4f}")

print(f"\n✅ Best Model: {best_name}")
print(f"   ROC-AUC Score  : {best['roc_auc']:.4f}")
print(f"   Avg Precision  : {best['avg_precision']:.4f}")
print("\nAll plots saved to current directory.")
print("=" * 60)
# ============================================================
# SAVE BEST MODEL
# ============================================================

import joblib

joblib.dump(best["model"], "models/best_model.pkl")

print("\nBest model saved successfully!")