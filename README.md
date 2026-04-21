# Cryptocurrency Transaction Risk Analysis

This project analyzes blockchain cryptocurrency transactions to predict transaction risk using machine learning models. The goal is to identify potentially risky transactions based on various features such as transaction amount, sender/receiver information, and other blockchain metrics.

## Dataset

The dataset used is `Blockchain_Cryptocurrency_Transaction_Risk_Dataset.csv`, which contains cryptocurrency transaction data with the following key features:
- Transaction ID
- Wallet ID
- Transaction amount
- Sender/Receiver details
- Risk label (target variable: 0 for low risk, 1 for high risk)

## Models Implemented

The project evaluates several machine learning models for risk prediction:

1. **Logistic Regression** - A linear model for binary classification
2. **Random Forest** - An ensemble method using multiple decision trees
3. **XGBoost** - A gradient boosting framework
4. **Support Vector Machine (SVM)** - A kernel-based classification method

## Features

- Data preprocessing and feature scaling
- Handling imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique)
- Model training and evaluation
- Performance metrics: Accuracy, R² Score, ROC-AUC
- Confusion matrices and classification reports
- Feature importance analysis (for Random Forest)
- Model comparison and visualization

## Requirements

To run this project, you'll need the following Python libraries:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Saee25/Cryptocurrency-transaction-risk.git
cd Cryptocurrency-transaction-risk
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the dataset file `Blockchain_Cryptocurrency_Transaction_Risk_Dataset.csv` is in the same directory as the notebook.

2. Open the Jupyter notebook `Blockchain_Cryptocurrency_Transaction_Risk.ipynb`.

3. Run the cells in order to:
   - Load and preprocess the data
   - Apply SMOTE for balancing
   - Train and evaluate each model
   - Generate comparison plots and save results

## Results

The notebook outputs a comparison of model performances, including accuracy and ROC-AUC scores. Results are saved to `model_comparison_results.csv`.

Key findings include:
- Model accuracy scores
- ROC-AUC performance metrics
- Feature importance rankings
- Confusion matrices for each model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
