# 🛰️ Space Traffic Management: LEO Orbit Prediction with XGBoost, LightGBM, and FT-Transformer

<div align="center">

![Project Banner](https://img.shields.io/badge/Project%20Type-Data%20Science%2FML-blue.svg?style=for-the-badge)
[![GitHub stars](https://img.shields.io/github/stars/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=github)](https://github.com/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=git)](https://github.com/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/network)
[![GitHub issues](https://img.shields.io/github/issues/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=github)](https://github.com/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/issues)
[![GitHub license](https://img.shields.io/github/license/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge)](LICENSE)

**Using Machine Learning to predict satellite collisions and manage space traffic in Low Earth Orbit (LEO).**

</div>

---

## 📖 Overview

Low Earth Orbit (LEO) is getting crowded. With thousands of satellites and pieces of debris flying around, operators get thousands of "collision alerts" every day. Most of these are false alarms, but checking them takes time and fuel.

**The Goal:** We built a Machine Learning system to look at Conjunction Data Messages (CDMs)—the reports that say "two things might hit each other"—and predict which ones are **actual high-risk events**.

**The Solution:** We processed over **574,000 data points** using three powerful models:
1.  **FT-Transformer:** A deep learning model designed specifically for data tables.
2.  **XGBoost:** A fast, tree-based model.
3.  **LightGBM:** Another efficient tree-based model.

We also created an **Ensemble** (a combination of models) to make the system more stable and reliable.

## 🧠 How It Works (The Technical Details)

We didn't just throw data at a model. We carefully engineered the system to handle the specific problems of space data.

### 1. Handling "Leakage"
Some data in the reports (like `cdmPc` - Probability of Collision) gives away the answer too easily. To make sure our models are actually learning useful patterns and not just cheating:
* We trained **"With-Leak"** models (using all data).
* We trained **"No-Leak"** models (hiding the obvious answers to test real learning).

### 2. Feature Engineering
We created new data points to help the models understand the physics better:
* **Log Probability:** Smoothed out extreme numbers.
* **Inverse Miss Distance:** Gave more importance to objects passing very close to each other.
* **Time Buckets:** Grouped events by how many hours were left until the potential crash.

### 3. The "Safety First" Threshold
In space, missing a crash is much worse than a false alarm. However, too many false alarms are annoying.
* We used a tool called **Optuna** to tune our models.
* **The Rule:** We set the model to find as many risks as possible (**Recall**), but it was *not allowed* to let its accuracy (**Precision**) drop below **50%**.
* This ensures that if the model warns you, there is at least a 50/50 chance it is a serious threat.

---

## ✨ Key Features

-   **Deep Learning for Tables:** Uses **FT-Transformer**, an attention-based neural network that treats spreadsheet data like language to find complex patterns.
-   **Smart Ensemble:** Combines 4 XGBoost models and 4 LightGBM models. If one model makes a mistake, the others balance it out.
-   **Imbalance Handling:** The dataset has very few actual crash risks compared to safe events. We used special weighting techniques so the model doesn't just guess "Safe" every time.
-   **Reproducible Pipeline:** From cleaning the messy raw data to training the final model, everything is scripted.
-   **Pre-trained Weights:** Includes `ft_transformer.pth` so you don't have to retrain the neural network from scratch.

---

## 🛠️ Tech Stack

**Languages & Core:**
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**Machine Learning:**
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-005101?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4169E1?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)

**Data Processing & Visualization:**
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Combined-blue)](https://optuna.org/)

---

## 🚀 Quick Start

### Prerequisites
* Python 3.8 or higher.
* Pip (Python package installer).

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer.git](https://github.com/NexusIITJ/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer.git)
    cd Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer
    ```

2.  **Create a virtual environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn optuna
    ```

4.  **Prepare the Data**
    * The raw data is in `sa-competition-files.zip`.
    * Unzip this file into a folder named `data/`.
    ```bash
    mkdir -p data
    unzip sa-competition-files.zip -d data/
    ```

---

## 📁 Project Structure

```text
.
├── IGOM_ML_LightGBM_Colab.ipynb   # The main notebook with experiments
├── ft_transformer.pth             # Saved weights for the Deep Learning model
├── predict_XGB.py                 # Script to run predictions with XGBoost
├── predict_LGBM.py                # Script to run predictions with LightGBM
├── predict_FTTransformer.py       # Script to run predictions with FT-Transformer
├── src/                           # Source code for preprocessing and utils
├── data/                          # Folder for your dataset (extract zip here)
├── results/                       # Where metrics and graphs are saved
└── outputs/                       # Where model predictions are saved
```
## 📊 Results Summary


-   **Pre-trained FT-Transformer:** The `ft_transformer.pth` file contains pre-trained weights for the FT-Transformer model. This allows for immediate loading and use for inference, or as a starting point for further fine-tuning.
-   **Model Storage (`models/`):** This directory is designated for saving trained instances of XGBoost, LightGBM, and FT-Transformer after experimentation.
-   **Output Data (`outputs/`):** Expected to contain generated predictions, processed intermediate datasets, or other files produced during the notebook execution.
-   **Experiment Results (`results/`):** This directory is where evaluation metrics, comparative plots, and other quantitative outcomes of the experiments should be stored.

**  FT-Transformer:** Showed incredible accuracy, achieving a perfect score on some test sets (Precision 1.0, Recall 1.0) when using the full feature set.
**Ensemble (XGBoost + LightGBM):** Provided the most stable results. It reduced noise and successfully kept Precision above 50% while finding all the high-risk events.
**Conclusion:** Transformer models are very promising for space data, and Ensembles are excellent for operational safety.

## 🤝 Contributing

We welcome contributions to further enhance this Space Traffic Management project! Whether you aim to improve model performance, integrate new algorithms, refine data processing, or enhance documentation, your efforts are appreciated. Please refer to these general guidelines:

1.  **Fork** the repository.
2.  **Clone** your forked repository to your local machine.
3.  Create a new **branch** (`git checkout -b feature/your-feature-name`).
4.  Make your changes, ensuring code is well-commented and clear.
5.  **Commit** your changes.
6.  **Push** to the branch.
7.  Open a **Pull Request**.

## 📜 Citation & License

Citation:

*[1] S. NexusIITJ, “Space Traffic Management – LEO Orbit – XGBoost, LightGBM and FT‑Transformer,” GitHub Repository, 2025*.

**Disclaimer:** © 2025 Space Debris Conference 2026. The statements and opinions expressed in this paper are solely those of the authors and do not necessarily reflect the views of the conference organizers, affiliated institutions, or the publisher.

## 🙏 Acknowledgments

-   To the creators and maintainers of **XGBoost**, **LightGBM**, **PyTorch**, **scikit-learn**, **Pandas**, and **NumPy** for providing powerful open-source tools that are foundational to this project.
-   The broader scientific community and space agencies for their ongoing work in advancing Space Traffic Management.


<div align="center">

**⭐  If you find this useful for Space Safety, please give us a star!**

Made by the **Nexus** Astronomy and Space-Tech Club of IIT Jodhpur.

</div>