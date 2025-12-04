# 🐎 Horse Survival Prediction — Kaggle Competition (1st Place)

A machine learning project that predicts the health outcomes of horses (lived, died, euthanized) using advanced feature engineering, LightGBM with Optuna tuning, and Stratified K-Fold cross-validation.  
This model ranked **1st on both the public and private leaderboards**.

---

## 🏆 Competition Results (Final Standing)

| Leaderboard | Score | Rank |
|------------|--------|------|
| **Public LB (20%)** | **0.8475** | 🥇 1st |
| **Private LB (80%)** | **0.7742** | 🥇 1st |

---

## 📌 Problem Overview

Given various **clinical features** of horses (heart rate, rectal temperature, protein levels, respiratory rate, etc.), the task is to predict the **final health outcome**:

- `lived`
- `died`
- `euthanized`

This is a **multiclass classification** problem evaluated using **Micro-F1 Score**.

---

## 🗂 Dataset

The dataset consists of three main files:

- `train.csv` –  (Training data with labels)
- `test.csv` –  (Test data for final submission)
- `horse.csv` –  (Additional mixed-source dataset used for boosting training size)

### ✔ Target Column  
- `outcome` → multiclass (lived / died / euthanized)

### ✔ Data Issues Handled  
- Missing numeric values → **median imputation**  
- Missing categorical values → **mode imputation**  
- Duplicates removed  
- Several noisy columns removed (e.g., `lesion_3`, `id`)

---

## 🔧 Technologies Used

- Python  
- LightGBM  
- Optuna (Hyperparameter Tuning)  
- Pandas / NumPy  
- Scikit-Learn  
- Matplotlib  

---

## 🧪 Data Preparation

### ✔ Merging datasets
train.csv + horse.csv → combined dataset

### ✔ Missing Value Handling
- Numeric → median  
- Categorical → most frequent value  

### ✔ Feature Cleanup
Dropped columns:
- `id`
- `lesion_3`

### ✔ Train/Test Split
Used during CV only. Final submission trained on full data.

---

## 🧠 Feature Engineering

Several domain-based engineered features were added:

### **1. Temperature Deviation**
abs(rectal_temp - 37.8)  

### **2. Pulse Categories**
- low  
- normal  
- elevated  
- high  

### **3. Total Protein Levels**
- low  
- normal  
- elevated  
- high  

### **4. Packed Cell Volume Categories**
- low  
- normal  
- high  

### **5. Interaction Features**
pulse * respiratory_rate

### **6. Convert categorical features to strings for encoding**

This allows OrdinalEncoder to handle unseen values safely.

---

## 🏗 Encoding Categorical Features

Used **OrdinalEncoder**, with:
- `handle_unknown='use_encoded_value'`
- `unknown_value=-1`

So the model does NOT crash on new categories in test data.

---

## 🚀 Model Used — LightGBM

LightGBM was chosen because:
- It handles multiclass problems easily  
- Works extremely well with structured/tabular data  
- Supports missing values internally  
- Very fast with high accuracy  

### **Objective**
objective = 'multiclass'
num_class = 3

### **Evaluation Metric**
f1_micro

---

## 🎛 Hyperparameter Tuning — Optuna

The following parameters were optimized:

| Parameter | Meaning | Range |
|----------|----------|--------|
| learning_rate | Boosting step | 0.01 – 0.2 |
| num_leaves | Tree complexity | 20 – 60 |
| max_depth | Max tree depth | 3 – 8 |
| min_child_samples | Min data in leaf | 10 – 50 |
| min_child_weight | Gradient threshold | 1e-5 – 0.1 |
| subsample | Row sampling | 0.6 – 1.0 |
| colsample_bytree | Feature sampling | 0.6 – 1.0 |
| lambda_l1 | L1 regularization | 1e-5 – 5.0 |
| lambda_l2 | L2 regularization | 1e-5 – 5.0 |

Optuna helped the model reach **state-of-the-art accuracy**.

---

## 🔁 Cross-Validation

Used **5-Fold Stratified K-Fold** to maintain class balance.

### 📊 Fold Results (Micro-F1)

| Fold | Score |
|------|--------|
| 1 | 0.7818 |
| 2 | 0.7068 |
| 3 | 0.7394 |
| 4 | 0.7752 |
| 5 | 0.7647 |

📌 **Mean CV Micro-F1:** **0.7536**

You generated this visualization → `cv_scores.png`

---

## 📈 Feature Importance

The features contributing most to the model:

| Feature | Importance |
|---------|------------|
| lesion_1 | 701.4 |
| hospital_number | 669.8 |
| total_protein | 459.8 |
| packed_cell_volume | 457.4 |
| pulse | 413.4 |

Visualization file saved as:

👉 `feature_importance.png`

---

## 🧩 Final Model & Submission

Steps:

1. Trained LightGBM on full combined data using tuned hyperparameters  
2. Predicted outcomes for test data  
3. Converted numeric predictions back to labels using LabelEncoder  
4. Saved final results to `submission.csv`  

This submission achieved:

### 🥇 Public LB: **0.8475**  
### 🥇 Private LB: **0.7742** (Final Rank: **1st Place**)  

---

## 🗂 Project Structure

horse-survival-kaggle/
│── explore.py
│── model.py
│── train.csv
│── test.csv
│── horse.csv
│── submission.csv
│── sample_submission.csv
│── Horse Outcome Notes.docx
│
├── results/
│ ├── feature_importance.png
│ ├── cv_scores.png
│ └── confusion_matrix.png (optional)
│
└── README.md

---

## 🚀 Future Improvements

- Add SHAP values for deeper interpretability  
- Try CatBoost for categorical handling  
- Try XGBoost & ensemble models  
- Apply advanced hyperparameter sweeps  
- Explore automatic feature selection  

---

## 📬 Contact

**Arnav Saxena**  
🔗 LinkedIn: https://www.linkedin.com/in/arnav-saxena-a9a217367  
📧 Email: **arnav12saxena@gmail.com**

---

