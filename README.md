# 🛰️ Space Traffic Management: LEO Orbit Prediction with XGBoost, LightGBM, and FT-Transformer

<div align="center">

![Project Banner](https://img.shields.io/badge/Project%20Type-Data%20Science%2FML-blue.svg?style=for-the-badge)
[![GitHub stars](https://img.shields.io/github/stars/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=github)](https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=git)](https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/network)
[![GitHub issues](https://img.shields.io/github/issues/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=github)](https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/issues)
[![GitHub license](https://img.shields.io/github/license/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge)](LICENSE)

**Leveraging advanced machine learning for robust prediction and management of Low Earth Orbit (LEO) space traffic.**

</div>

## 📖 Overview

This project delves into the critical domain of Space Traffic Management (STM) in Low Earth Orbit (LEO) by implementing and comparing advanced machine learning models. With the exponential growth of satellite constellations and orbital debris, accurate prediction of orbital trajectories and collision risks is vital for ensuring the long-term sustainability and safety of space operations.

This repository provides a comprehensive Jupyter notebook that showcases the application of popular gradient boosting algorithms, **XGBoost** and **LightGBM**, alongside the innovative deep learning architecture, **FT-Transformer (Feature-Tokenization Transformer)**, for handling tabular data in this complex space domain. The goal is to demonstrate the efficacy of these models in extracting actionable insights and making reliable predictions for LEO traffic.

## ✨ Features

-   **Multi-Model Comparative Analysis:** Implements and compares the predictive performance of XGBoost, LightGBM, and FT-Transformer models for LEO STM.
-   **Targeted LEO Focus:** Addresses the unique challenges and high-density environment characteristic of Low Earth Orbit.
-   **Data Processing Pipeline:** Includes stages for loading raw competition data, essential preprocessing, and feature engineering to prepare data for diverse ML architectures.
-   **Robust Model Training & Evaluation:** Demonstrates the full lifecycle of model development, including training, hyperparameter tuning insights, and rigorous evaluation using appropriate metrics within a reproducible notebook.
-   **Deep Learning for Tabular Data:** Explores the application of the FT-Transformer, a state-of-the-art neural network, for superior feature learning from structured data.
-   **Pre-trained Model Integration:** Includes pre-trained weights (`ft_transformer.pth`) for the FT-Transformer, allowing for direct evaluation or fine-tuning.

## 🛠️ Tech Stack

**Primary Language:**
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**Machine Learning & Data Science Libraries:**
[![XGBoost](https://img.shields.io/badge/XGBoost-005101?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4169E1?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-BA3B0C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-328699?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)

**Development Tools:**
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

## 🚀 Quick Start

To set up this project and run the orbital traffic management experiments, follow these instructions.

### Prerequisites

-   **Python 3.8+** (or newer, as recommended by the ML libraries)
-   **pip** (Python package installer, usually bundled with Python)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer.git
    cd Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer
    ```

2.  **Create and activate a virtual environment (recommended)**
    Using a virtual environment helps manage dependencies for different projects without conflicts.
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    Since a `requirements.txt` file is not provided, please install the necessary libraries manually. It is highly recommended to generate a `requirements.txt` (e.g., `pip freeze > requirements.txt`) after setting up your environment for future reproducibility.
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn jupyterlab
    ```
    *Note: If you are using Google Colab, many of these libraries (e.g., NumPy, Pandas, Scikit-learn, PyTorch) are often pre-installed. You might only need to install `xgboost` and `lightgbm` if they are not already available.*

4.  **Prepare the dataset**
    The raw dataset for the competition is provided as `sa-competition-files.zip`. You need to unzip it to access the data. A common practice is to extract it into a dedicated `data/` directory.
    ```bash
    mkdir -p data # Create a 'data' directory if it doesn't exist
    unzip sa-competition-files.zip -d data/
    ```

5.  **Run the Jupyter Notebook**
    The core analysis and model implementations are within the Jupyter Notebook. You can run it locally or leverage Google Colab for cloud-based execution.

    *   **Option A: Google Colab**
        1.  Go to [Google Colab](https://colab.research.google.com/).
        2.  Click `File` -> `Upload notebook` and upload `IGOM_ML_LightGBM_Colab.ipynb`.
        3.  Ensure you also upload the `ft_transformer.pth` file and follow instructions within the notebook to unzip `sa-competition-files.zip` if you didn't do it locally.
        4.  Run all cells in the notebook.

    *   **Option B: Local Jupyter Environment**
        1.  Ensure you have `jupyterlab` installed (it was included in the `pip install` command).
        2.  Start Jupyter Lab:
            ```bash
            jupyter lab
            ```
        3.  Your web browser will open, navigate to and open `IGOM_ML_LightGBM_Colab.ipynb`.
        4.  Execute the cells sequentially to run the experiments.

## 📁 Project Structure

```
.
├── .gitignore                          # Standard Git ignore file
├── IGOM_ML_LightGBM_Colab.ipynb      # Main Jupyter notebook containing ML experiments and analysis
├── README.md                           # The project's README file
├── ft_transformer.pth                  # Pre-trained weights for the FT-Transformer model
├── models/                             # (Empty) Directory intended for saving trained models (e.g., .pkl, .pt)
├── outputs/                            # (Empty) Directory intended for generated outputs like predictions or intermediate data
├── results/                            # (Empty) Directory intended for experimental results, metrics, and plots
├── sa-competition-files.zip            # Zipped raw dataset for the space traffic management competition
├── src/                                # (Empty) Placeholder for source code, utility scripts, or custom modules
└── testing.py                          # (Empty) Placeholder for unit/integration tests
```

## ⚙️ Configuration

All model configurations, hyperparameter settings, data preprocessing steps, and experimental parameters are defined directly within the `IGOM_ML_LightGBM_Colab.ipynb` notebook. There are no external configuration files (e.g., `.env`, `config.yaml`) or specific environment variables explicitly detected and used by the project.

## 📊 Results & Models

-   **Pre-trained FT-Transformer:** The `ft_transformer.pth` file contains pre-trained weights for the FT-Transformer model. This allows for immediate loading and use for inference, or as a starting point for further fine-tuning.
-   **Model Storage (`models/`):** This directory is designated for saving trained instances of XGBoost, LightGBM, and FT-Transformer after experimentation.
-   **Output Data (`outputs/`):** Expected to contain generated predictions, processed intermediate datasets, or other files produced during the notebook execution.
-   **Experiment Results (`results/`):** This directory is where evaluation metrics, comparative plots, and other quantitative outcomes of the experiments should be stored.

The `IGOM_ML_LightGBM_Colab.ipynb` notebook details the process to generate, analyze, and store these outputs.

## 🤝 Contributing

We welcome contributions to further enhance this Space Traffic Management project! Whether you aim to improve model performance, integrate new algorithms, refine data processing, or enhance documentation, your efforts are appreciated. Please refer to these general guidelines:

1.  **Fork** this repository.
2.  **Clone** your forked repository to your local machine.
3.  Create a new **branch** (`git checkout -b feature/your-feature-name`).
4.  Make your changes, ensuring code is well-commented and clear.
5.  **Commit** your changes (`git commit -m 'feat: Add new feature X'`).
6.  **Push** to your branch (`git push origin feature/your-feature-name`).
7.  Open a **Pull Request** to the `main` branch of this repository, describing your contributions.

## 📄 License

This project is licensed under the [LICENSE_NAME](LICENSE). Please see the `LICENSE` file for full details.
*(Note: A `LICENSE` file was not found in the repository. Please add a `LICENSE` file to specify the terms under which your project can be used, distributed, and modified.)*

## 🙏 Acknowledgments

-   To the creators and maintainers of **XGBoost**, **LightGBM**, **PyTorch**, **scikit-learn**, **Pandas**, and **NumPy** for providing powerful open-source tools that are foundational to this project.
-   The broader scientific community and space agencies for their ongoing work in advancing Space Traffic Management.

## 📞 Support & Contact

-   🐛 Issues & Bug Reports: Please use the [GitHub Issues](https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/issues) section.
-   👤 Author: ShivJee-Yadav ([GitHub Profile](https://github.com/ShivJee-Yadav))

---

<div align="center">

**⭐ If this project aids your understanding or work in Space Traffic Management, please consider starring the repository!**

Made with ❤️ by ShivJee-Yadav

</div>