# SDC
<!-- Intall first 
pip install xgboost scikit-learn pandas
Data 
Slingshot Seradata
SlingShot Aerospace 
 www.seradata.com -->

# 🚀 Conjunction Risk Prediction using Machine Learning
*A complete machine‑learning pipeline for predicting high‑risk satellite conjunctions using CDM data.*

---

## 📌 Overview

This project builds an end‑to‑end system to identify **high‑risk conjunction events** in Low Earth Orbit (LEO).  
As the number of satellites and debris increases, operators receive thousands of Conjunction Data Messages (CDMs), but only a small fraction represent real danger.  
This project aims to:

- Clean and standardize CDM data  
- Remove leakage features  
- Train multiple ML models  
- Compare their performance  
- Build a recall‑focused ensemble for safer screening  

The final system helps operators **prioritize risky encounters** without replacing human judgment.

---

## 📂 Project Structure
 


---

## 🧹 Data Processing

We combine **six CDM files** into a single dataset of **574,289 rows**.  
Key preprocessing steps include:

- Standardizing column names  
- Converting condition flags into boolean features  
- Label‑encoding categorical fields  
- Adding `hours_to_tca`  
- Removing leakage features for no‑leak variants  
- Creating engineered features (distance ratios, uncertainty indicators, etc.)

The final featured dataset contains **33 columns**.

---

## 🤖 Models Used

### **1. FT‑Transformer**
- Works directly on tabular data  
- Uses embeddings + attention  
- Predicts both Pc and HighRisk probability  
- Very strong performance on original data  

### **2. XGBoost (4 variants)**
- Original  
- Featured  
- No‑Leak  
- No‑Leak + Featured  
- Hyperparameters tuned using Optuna  

### **3. LightGBM (4 variants)**
- Same four variants as XGBoost  
- Fast and efficient baseline  

### **4. Ensemble**
- Combines 4 XGBoost + 4 LightGBM models  
- Weighted using Optuna  
- Designed for **maximum recall** with acceptable precision  

---

## 📊 Evaluation

Each model is evaluated using:

- Recall  
- Precision  
- F1‑Score  
- Accuracy  
- AUC‑ROC  
- AUC‑PR  
- Confusion matrices  
- Threshold scanning (to maximize recall with precision ≥ 0.50)

All results are saved as JSON for reproducibility.

---

## 📝 Key Findings

- Models **without leakage** struggle but still learn useful patterns.  
- Engineered features significantly improve performance.  
- Models **with leakage** perform almost perfectly because they see the fields used to define the label.  
- FT‑Transformer is extremely strong on original data.  
- The ensemble achieves **100% recall**, making it suitable for safety‑critical screening.

---

## 🛠️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt


python src/preprocessing.py