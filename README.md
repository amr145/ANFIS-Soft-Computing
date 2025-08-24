# ANFIS - Soft Computing (Diabetes Prediction)

## Overview
This project implements an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** model to predict **diabetes** based on patient medical data.  
It combines the strengths of **fuzzy logic** and **neural networks** to handle uncertainty and learn from data effectively.  

## Features
- Custom implementation of ANFIS in Python  
- Triangular membership functions  
- Rule-based fuzzy inference system  
- Gradient descent training for parameter optimization  
- Evaluation using **Accuracy** and **AUC-ROC** metrics  

## Tech Stack
- Python  
- NumPy & Pandas  
- Scikit-learn  
- Matplotlib (for optional visualizations)  

## Dataset
- **File**: `diabetes_dataset_with_notes.csv`  
- Preprocessing steps:
  - Dropped irrelevant columns (`location`, `clinical_notes`)  
  - Encoded categorical features (`gender`, `smoking_history`)  
  - Handled missing values  
  - Normalized features with **MinMaxScaler**  
  - Train/Test split (80/20)  

## Project Structure
ANFIS-Soft-Computing/
├── diabetes_dataset_with_notes.csv
├── anfis_diabetes.py
└── README.md

## How to Run
1. Clone the repository:
   git clone https://github.com/amr145/ANFIS-Soft-Computing.git
   cd ANFIS-Soft-Computing

2. Install dependencies:
   pip install -r requirements.txt

3. Run the model:
   python anfis_diabetes.py

## Example Output
```
Epoch 0: Train Accuracy = 0.72, Test Accuracy = 0.70
Epoch 10: Train Accuracy = 0.81, Test Accuracy = 0.78
...
=== Final Evaluation ===
Accuracy: 82.50%
AUC-ROC Score: 0.8734
```

## Future Improvements
- Implement Gaussian membership functions  
- Add more datasets for robustness  
- Hyperparameter tuning for better accuracy  
- Deploy as a web API for real-time predictions  
