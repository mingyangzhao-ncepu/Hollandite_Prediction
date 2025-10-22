# Hollandite_Prediction

This repository contains all code and datasets used in the manuscript:

**Physically Interpretable Models for Predicting Lattice Parameters, Tunnel Geometry, and Symmetry in Hollandite-Type Materials**

## Repository Structure

The repository consists of two main folders: `Code` and `Data`.

---

### 📁 Code

This folder contains all scripts for model development and evaluation. It is organized into two subdirectories:

#### 🔹 ML  
- Contains scripts for **machine learning models**, including **Artificial Neural Networks (ANN)** and **Support Vector Regression (SVR)**.
- Written in **Python**.
- Can be executed using any Python-supported IDE, such as **PyCharm** or **Jupyter Notebook**.

#### 🔹 Regression  
- Contains scripts for interpretable regression models, divided into three subfolders:
  - `MLR` – Multivariate Linear Regression
  - `MPR` – Multivariate Polynomial Regression (degree ≤ 2)
  - `Symmetry` – Regression-based symmetry classification model
- All scripts in this folder are written in **MATLAB**.

---

### 📁 Data

This folder contains the spreadsheets used for model training and prediction:

- `Hollandite_data.xlsx`  
  Used by both ML and regression models to predict **lattice constants a and c**.

- `Hollandite_data_dd.xlsx`  
  Used by both ML and regression models to predict **tunnel bottleneck size (d–d)**.

- `Hollandite_data_Symmetry.xlsx`  
  Used by regression models to classify **symmetry (tetragonal vs monoclinic)**.

---

## Usage

1. Select the appropriate script from either the `ML` or `Regression` folder based on your desired model type.
2. Load the corresponding dataset from the `Data` folder.
3. Run the model using Python or MATLAB depending on the script language.

---

## Citation

If you use this repository, please cite our paper:

> Zhao, M., Feng, Z., Chen, X., et al. (2025). Physically Interpretable Models for Predicting Lattice Parameters, Tunnel Geometry, and Symmetry in Hollandite-Type Materials. *Computational Materials Science*, [DOI].
