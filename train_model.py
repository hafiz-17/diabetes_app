"""
Diabetes Prediction Model Training Script
Generates trained model + performance graphs
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
from sklearn.pipeline import Pipeline

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, "diabetes_prediction_dataset.csv")
MODEL_DIR   = os.path.join(BASE, "models")
STATIC_IMG  = os.path.join(BASE, "static", "images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_IMG, exist_ok=True)

# ── Colour palette ───────────────────────────────────────────────────────────
PALETTE = {
    "primary":  "#667eea",
    "accent":   "#f093fb",
    "success":  "#43e97b",
    "danger":   "#f5576c",
    "warning":  "#fda085",
    "dark":     "#1a1a2e",
    "card":     "#16213e",
}
plt.rcParams.update({
    "figure.facecolor": PALETTE["dark"],
    "axes.facecolor":   PALETTE["card"],
    "axes.edgecolor":   "#ffffff33",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#ffffff15",
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
})

print("=" * 60)
print("  DIABETES PREDICTION MODEL TRAINING")
print("=" * 60)

# ── 1. Load & pre-process ────────────────────────────────────────────────────
print("\n[1/6] Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"  Diabetic: {df['diabetes'].sum():,}  |  Non-diabetic: {(df['diabetes']==0).sum():,}")

# Encode categoricals
le_gender  = LabelEncoder()
le_smoking = LabelEncoder()
df["gender"]          = le_gender.fit_transform(df["gender"])
df["smoking_history"] = le_smoking.fit_transform(df["smoking_history"])

FEATURES = ["gender","age","hypertension","heart_disease",
            "smoking_history","bmi","HbA1c_level","blood_glucose_level"]
TARGET   = "diabetes"

X = df[FEATURES]
y = df[TARGET]

# ── 2. Train / test split ────────────────────────────────────────────────────
print("\n[2/6] Splitting data (80/20 stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 3. Train model ───────────────────────────────────────────────────────────
print("\n[3/6] Training Gradient Boosting Classifier …")
model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.08, max_depth=5,
    min_samples_split=10, subsample=0.8, random_state=42)

# Track training progress (staged predict)
train_scores, test_scores = [], []
rf_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.08, max_depth=5,
    min_samples_split=10, subsample=0.8, random_state=42,
    warm_start=False)
rf_model.fit(X_train, y_train)

for i, y_pred_stage in enumerate(rf_model.staged_predict(X_train)):
    train_scores.append(accuracy_score(y_train, y_pred_stage))
for i, y_pred_stage in enumerate(rf_model.staged_predict(X_test)):
    test_scores.append(accuracy_score(y_test, y_pred_stage))

model = rf_model
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"  Test Accuracy: {acc:.4f}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print(f"  CV Accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 4. Save artefacts ────────────────────────────────────────────────────────
print("\n[4/6] Saving model & encoders …")
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(model,     os.path.join(MODEL_DIR, "diabetes_model.pkl"))
joblib.dump(scaler,    os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le_gender, os.path.join(MODEL_DIR, "le_gender.pkl"))
joblib.dump(le_smoking,os.path.join(MODEL_DIR, "le_smoking.pkl"))
joblib.dump(FEATURES,  os.path.join(MODEL_DIR, "features.pkl"))
print("  Saved to /models/")

# ── 5. Generate graphs ───────────────────────────────────────────────────────
print("\n[5/6] Generating performance graphs …")

def save_fig(name):
    path = os.path.join(STATIC_IMG, name)
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor=PALETTE["dark"])
    plt.close()
    print(f"  Saved: {name}")

# ─ 5a. Accuracy / Loss Curve ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Learning Curves", fontsize=16, fontweight="bold", color="white", y=1.02)

epochs = range(1, len(train_scores)+1)
ax = axes[0]
ax.plot(epochs, train_scores, color=PALETTE["primary"],  lw=2, label="Train Accuracy")
ax.plot(epochs, test_scores,  color=PALETTE["accent"],   lw=2, label="Test Accuracy",  linestyle="--")
ax.fill_between(epochs, train_scores, test_scores, alpha=0.08, color=PALETTE["primary"])
ax.set_xlabel("Estimators"); ax.set_ylabel("Accuracy")
ax.set_title("Accuracy over Estimators", fontweight="bold")
ax.legend(facecolor="#ffffff10"); ax.grid(True)

# Loss proxy (1-accuracy)
train_loss = [1-s for s in train_scores]
test_loss  = [1-s for s in test_scores]
ax = axes[1]
ax.plot(epochs, train_loss, color=PALETTE["warning"], lw=2, label="Train Loss")
ax.plot(epochs, test_loss,  color=PALETTE["danger"],  lw=2, label="Test Loss", linestyle="--")
ax.fill_between(epochs, train_loss, test_loss, alpha=0.08, color=PALETTE["danger"])
ax.set_xlabel("Estimators"); ax.set_ylabel("Loss (1 - Accuracy)")
ax.set_title("Loss over Estimators", fontweight="bold")
ax.legend(facecolor="#ffffff10"); ax.grid(True)

plt.tight_layout()
save_fig("learning_curves.png")

# ─ 5b. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
            xticklabels=["No Diabetes","Diabetes"],
            yticklabels=["No Diabetes","Diabetes"],
            linewidths=2, linecolor="#1a1a2e",
            annot_kws={"size": 18, "weight": "bold"}, ax=ax)
ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold", pad=15)
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout()
save_fig("confusion_matrix.png")

# ─ 5c. ROC Curve ─────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5,
        label=f"ROC Curve  (AUC = {roc_auc:.4f})")
ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["primary"])
ax.plot([0,1],[0,1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
ax.legend(facecolor="#ffffff10"); ax.grid(True)
plt.tight_layout()
save_fig("roc_curve.png")

# ─ 5d. Precision-Recall Curve ────────────────────────────────────────────────
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, color=PALETTE["accent"], lw=2.5,
        label=f"PR Curve  (AUC = {pr_auc:.4f})")
ax.fill_between(recall, precision, alpha=0.15, color=PALETTE["accent"])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold")
ax.legend(facecolor="#ffffff10"); ax.grid(True)
plt.tight_layout()
save_fig("pr_curve.png")

# ─ 5e. Feature Importance ────────────────────────────────────────────────────
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)

nice_names = {
    "blood_glucose_level": "Blood Glucose Level",
    "HbA1c_level":         "HbA1c Level",
    "bmi":                 "BMI",
    "age":                 "Age",
    "hypertension":        "Hypertension",
    "heart_disease":       "Heart Disease",
    "smoking_history":     "Smoking History",
    "gender":              "Gender",
}
feat_df["Label"] = feat_df["Feature"].map(nice_names)

colors = [PALETTE["danger"] if v > 0.15
          else PALETTE["warning"] if v > 0.07
          else PALETTE["primary"]
          for v in feat_df["Importance"]]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feat_df["Label"], feat_df["Importance"],
               color=colors, edgecolor="#ffffff20", height=0.6)
for bar, val in zip(bars, feat_df["Importance"]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", color="white", fontsize=10, fontweight="bold")

high_p = mpatches.Patch(color=PALETTE["danger"],  label="High Impact  (>15%)")
med_p  = mpatches.Patch(color=PALETTE["warning"], label="Medium Impact (7–15%)")
low_p  = mpatches.Patch(color=PALETTE["primary"], label="Low Impact   (<7%)")
ax.legend(handles=[high_p, med_p, low_p], facecolor="#ffffff10", loc="lower right")
ax.set_title("Feature Importance", fontsize=15, fontweight="bold")
ax.set_xlabel("Importance Score"); ax.grid(True, axis="x")
plt.tight_layout()
save_fig("feature_importance.png")

# ─ 5f. Class Distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["diabetes"].value_counts()
bars = ax.bar(["No Diabetes","Diabetes"], counts.values,
              color=[PALETTE["success"], PALETTE["danger"]],
              edgecolor="#ffffff20", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f"{val:,}", ha="center", fontsize=13, fontweight="bold", color="white")
ax.set_title("Dataset Class Distribution", fontsize=15, fontweight="bold")
ax.set_ylabel("Count"); ax.grid(True, axis="y")
plt.tight_layout()
save_fig("class_distribution.png")

# ─ 5g. CV Score Bar ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
folds = [f"Fold {i+1}" for i in range(5)]
bar_colors = [PALETTE["primary"] if s >= cv_scores.mean() else PALETTE["accent"] for s in cv_scores]
bars = ax.bar(folds, cv_scores, color=bar_colors, edgecolor="#ffffff20", width=0.5)
ax.axhline(cv_scores.mean(), color=PALETTE["warning"], lw=2, linestyle="--",
           label=f"Mean = {cv_scores.mean():.4f}")
for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", fontsize=10, fontweight="bold", color="white")
ax.set_ylim([cv_scores.min()-0.02, 1.0])
ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=15, fontweight="bold")
ax.set_ylabel("Accuracy"); ax.legend(facecolor="#ffffff10"); ax.grid(True, axis="y")
plt.tight_layout()
save_fig("cv_scores.png")


