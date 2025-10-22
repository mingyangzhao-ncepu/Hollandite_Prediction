
# Hollandite_Prediction

This repository contains the source code and data associated with the manuscript:
**"Physically Interpretable Models for Predicting Lattice Parameters, Tunnel Geometry, and Symmetry in Hollandite-Type Materials"**.

## Repository Structure

```
Hollandite_Prediction/
├── Code/
│   ├── ML/              # Python code for Artificial Neural Network (ANN) and Support Vector Regression (SVR)
│   └── Regression/      # MATLAB code for Multivariate Linear Regression (MLR), Polynomial Regression (MPR), and Symmetry classification
├── Data/
│   ├── Hollandite_data.xlsx               # Dataset for predicting lattice constants a and c
│   ├── Hollandite_data_dd.xlsx            # Dataset for predicting tunnel bottleneck size d-d
│   └── Hollandite_data_Symmetry.xlsx      # Dataset for predicting symmetry classification
```

## Code Description

- **Code/ML**: Contains Python scripts for ANN and SVR models. These can be executed using common Python IDEs such as PyCharm or Jupyter Notebook.
- **Code/Regression**: Contains MATLAB scripts for:
  - **MLR**: Multivariate Linear Regression for predicting lattice parameter a, c, and d-d.
  - **MPR**: Polynomial Regression (up to second degree) for predicting lattice parameter a, c, and d-d.
  - **Symmetry**: Regression model using a critical size descriptor (τ) to classify tetragonal vs. monoclinic structures.

## Data Description

- `Hollandite_data.xlsx`: Used for training and predicting lattice constants a and c.
- `Hollandite_data_dd.xlsx`: Used for tunnel bottleneck size (d-d) predictions.
- `Hollandite_data_Symmetry.xlsx`: Used for predicting crystal symmetry.

## Requirements

- Python ≥ 3.7 (with libraries: scikit-learn, numpy, pandas, matplotlib)
- MATLAB ≥ R2024a

## Citation

If you want to use the data and code in this repository, please contact the authors first. Permission and citation requirements will be provided accordingly.

## Contact

For inquiries, please contact: Mingyang Zhao (myzhao@ncepu.edu.cn); Zhenfeng Tong (zhenfeng_tong@ncepu.edu.cn)
