# 💳 Credit Card Fraud Detection using Machine Learning

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using advanced classification algorithms and class imbalance handling techniques.

---

## 🚀 Project Overview

Credit card fraud is one of the most significant challenges faced by financial institutions. Fraudulent transactions are extremely rare compared to legitimate transactions, making fraud detection a highly imbalanced classification problem.

This project builds a robust fraud detection pipeline that:

✅ Performs Exploratory Data Analysis (EDA)

✅ Handles severe class imbalance using SMOTE

✅ Trains and compares multiple Machine Learning models

✅ Evaluates performance using ROC-AUC and Precision-Recall metrics

✅ Identifies important fraud-related features

✅ Saves the best-performing model for future use

---

## 📊 Dataset

**Credit Card Fraud Detection Dataset**

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Statistics

| Metric                  | Value   |
| ----------------------- | ------- |
| Total Transactions      | 284,807 |
| Legitimate Transactions | 284,315 |
| Fraudulent Transactions | 492     |
| Fraud Percentage        | 0.172%  |

⚠️ The dataset is highly imbalanced, making fraud detection a challenging task.

---

## 🛠️ Tech Stack

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 📈 Matplotlib
* 🎨 Seaborn
* 🤖 Scikit-Learn
* ⚡ XGBoost
* ⚖️ SMOTE (Imbalanced-Learn)
* 💾 Joblib

---

## 🔍 Exploratory Data Analysis

The project performs:

* Class Distribution Analysis
* Transaction Amount Distribution
* Feature Correlation Analysis
* Fraud Pattern Investigation

### Generated Visualizations

📌 Class Distribution

📌 Transaction Amount Distribution

📌 Feature Correlation Analysis

---

## ⚙️ Data Preprocessing

### Feature Engineering

* Standardized `Amount` feature
* Standardized `Time` feature
* Removed original columns after scaling

### Handling Class Imbalance

Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic fraud samples and balance the training dataset.

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

| Model               |
| ------------------- |
| Logistic Regression |
| Decision Tree       |
| Random Forest       |
| Gradient Boosting   |
| XGBoost             |

---

## 📈 Model Evaluation

Evaluation Metrics:

* ROC-AUC Score
* Average Precision Score
* Classification Report
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

## 🏆 Best Model: XGBoost

### Performance Metrics

| Metric            | Score      |
| ----------------- | ---------- |
| ROC-AUC           | **0.9708** |
| Average Precision | **0.8200** |

### Confusion Matrix

| Actual / Predicted | Legitimate | Fraud |
| ------------------ | ---------- | ----- |
| Legitimate         | 85,257     | 38    |
| Fraud              | 29         | 119   |

✅ Very low False Positive Rate

✅ Strong Fraud Detection Capability

✅ Excellent ROC-AUC Performance

---

## 📊 Feature Importance Analysis

The project identifies the most influential features contributing to fraud detection using XGBoost feature importance scores.

Top important features include:

* V14
* V4
* V10
* V8
* V12
* V25
* V13

---

## 📂 Project Structure

```text
credit-card-fraud-detection/
│
├── app/
├── data/
│   └── creditcard.csv
│
├── models/
│   └── best_model.pkl
│
├── fraud_detection.py
├── requirements.txt
├── README.md
│
├── eda_plots.png
├── feature_correlation.png
├── feature_importance.png
├── model_comparison.png
└── best_model_analysis.png
```

---

## 🚀 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download Dataset

Download the dataset from Kaggle and place it inside:

```text
data/creditcard.csv
```

### 4️⃣ Run Project

```bash
python fraud_detection.py
```

---

## 📸 Project Outputs

The project automatically generates:

📈 `eda_plots.png`

📊 `feature_correlation.png`

📉 `model_comparison.png`

🎯 `best_model_analysis.png`

⭐ `feature_importance.png`

---

## 💾 Model Saving

The best-performing model is automatically saved as:

```text
models/best_model.pkl
```

This model can later be loaded for inference without retraining.

---

## 🎯 Future Improvements

* Hyperparameter Tuning (GridSearchCV)
* Real-Time Fraud Prediction System
* Flask/FastAPI Deployment
* Interactive Dashboard
* Model Monitoring Pipeline

---

## 👩‍💻 Author

### Ruthika Reddy

🎓 B.Tech Computer Science Engineering

📊 Data Science Enthusiast

🤖 Machine Learning & AI Explorer

💡 Passionate about solving real-world problems using data and intelligent systems
