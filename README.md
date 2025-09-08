# 🧬 Breast Cancer Diagnosis with PCA & Random Forest

![PCA](https://img.shields.io/badge/Dimensionality%20Reduction-PCA-blue)
![Random Forest](https://img.shields.io/badge/Classifier-Random%20Forest-green)
![License](https://img.shields.io/badge/Data-UCI%20ML%20Breast%20Cancer%20Wisconsin-orange)

---
## APP LINK
           https://rvfwfdq9nhkehpbh8tyx22.streamlit.app/
## 📂 Project Structure

```
PCA.ipynb
preprocessing.ipynb
visulas.ipynb
wdbc.data
```

---

## 🚀 Overview

This project demonstrates the use of **Principal Component Analysis (PCA)** for dimensionality reduction and **Random Forest** for classification on the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

- **preprocessing.ipynb**: Data loading, cleaning, and exploration.
- **visulas.ipynb**: Exploratory Data Analysis (EDA) and feature visualization.
- **PCA.ipynb**: PCA transformation, visualization, and classification.

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository
- **File**: `wdbc.data`
- **Features**: 30 numeric features computed from digitized images of breast mass
- **Target**: Diagnosis (`M` = Malignant, `B` = Benign)

---

## 🛠️ How to Run

1. **Clone the repository** and place all files in the same directory.
2. **Install dependencies** (Python 3.7+ recommended):

   ```sh
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Open the notebooks** in Jupyter or VS Code and run cells in order:

   - `preprocessing.ipynb`
   - `visulas.ipynb`
   - `PCA.ipynb`

---

## 🔍 Workflow

### 1. Data Preprocessing [`preprocessing.ipynb`](preprocessing.ipynb)
- Load and inspect data
- Handle missing values and duplicates
- Encode categorical variables

### 2. Visualization & EDA [`visulas.ipynb`](visulas.ipynb)
- Feature distributions
- Correlation heatmaps
- Skewness and outlier detection

### 3. PCA & Classification [`PCA.ipynb`](PCA.ipynb)
- Standardize features
- Apply PCA (2D & 3D)
- Visualize PCA results
- Train/test split
- Random Forest classification
- Evaluate with accuracy, precision, recall, F1, confusion matrix, ROC curve

---

## 📈 Example Results

- **Accuracy**: ~95%
- **Precision**: ~97%
- **Recall**: ~90%
- **F1 Score**: ~94%

![PCA 2D Scatter](https://img.icons8.com/color/48/000000/scatter-plot.png)
![Confusion Matrix](https://img.icons8.com/color/48/000000/confusion-matrix.png)
![ROC Curve](https://img.icons8.com/color/48/000000/roc-curve.png)

---

## 🤝 Credits

- Data: [UCI ML Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

---

## 📬 Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

> *Empowering early cancer detection with data
