import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🩺 Breast Cancer Diagnosis with PCA & Random Forest")
st.markdown("This app uses **Principal Component Analysis (PCA)** and a **Random Forest Classifier** "
            "to classify breast cancer tumors as **Benign (0)** or **Malignant (1)**.")

columns = [
    'ID', 'Diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

df = pd.read_csv("wdbc.data", header=None, names=columns)
df.drop("ID", axis=1, inplace=True)
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca3, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=["Benign","Malignant"], yticklabels=["Benign","Malignant"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig_cm)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc, ax = plt.subplots()
ax.plot(fpr, tpr, color="red", label=f"AUC = {roc_auc:.2f}")
ax.plot([0,1],[0,1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig_roc)

st.subheader("🖼 PCA Visualization")

fig2d, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(X_pca2[:,0], X_pca2[:,1], c=y, cmap="coolwarm", s=60, alpha=0.8)
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.2f}%)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.2f}%)")
ax.set_title("PCA 2D Scatter Plot")
st.pyplot(fig2d)

fig3d = plt.figure(figsize=(8,6))
ax = fig3d.add_subplot(111, projection="3d")
scatter = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=y, cmap="coolwarm", s=60, alpha=0.8)
ax.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.2f}%)")
ax.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]*100:.2f}%)")
ax.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.2f}%)")
ax.set_title("PCA 3D Scatter Plot")
st.pyplot(fig3d)

st.subheader("🔮 Predict New Sample")

input_features = []
for col in X.columns:
    val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
    input_features.append(val)

if st.button("Predict Diagnosis"):
    sample = np.array(input_features).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    sample_pca = pca3.transform(sample_scaled)
    pred = rf.predict(sample_pca)[0]
    prob = rf.predict_proba(sample_pca)[0][1]

    if pred == 1:
        st.error(f"⚠️ Prediction: **Malignant (Cancerous)** with probability {prob:.2f}")
    else:
        st.success(f"✅ Prediction: **Benign (Non-cancerous)** with probability {1-prob:.2f}")
