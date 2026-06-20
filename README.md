# UCI HAR Activity Recognition (Deep Learning + Streamlit)

A machine learning project that classifies human activities using sensor data from the **UCI HAR dataset**.  
The project includes data preprocessing, EDA, notebooks, PCA, deep learning model training, hyperparameter tuning, and a deployed Streamlit web app.

---

## Project Overview

This project predicts human activities such as:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

using 561 sensor features collected from smartphones.

---

## Tech Stack

- Python
- PyTorch (ANN model)
- Scikit-learn (PCA, metrics)
- Optuna (hyperparameter tuning)
- Pandas / NumPy
- Streamlit (web app)

---

## Workflow

1. Data preprocessing (UCI HAR dataset)
2. Exploratory Data Analysis (EDA)
3. Feature reduction using PCA (95% variance)
4. ANN model built using PyTorch
5. Hyperparameter tuning using Optuna
6. Final model training
7. Streamlit deployment

---

## Model Details

- Input Features: PCA reduced components
- Model: Fully Connected Neural Network (ANN)
- Optimizer: SGD
- Loss Function: CrossEntropyLoss
- Hyperparameter tuning: Optuna (50 trials)

---

## Project Structure
Project8/
│── streamlit_app.py
│── requirements.txt
│── README.md
│
├── src/
│ ├── model.py
│ ├── predict.py
│
├── models/
│ ├── pca.pkl
│ ├── ann_model.pth
│ ├── class_names.pkl
│ ├── best_params.pkl
│
├── notebooks/
│ ├── model01.ipynb
│ ├── model02.ipynb
│ ├── model03.ipynb


---

## How to Run Locally

### 1. Clone repo

git clone https://github.com/apaiglow/UCI-HAR-Activity-Classifier.git
cd UCI-HAR-Activity-Classifier

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run Streamlit app

streamlit run streamlit_app.py

---

## Dataset

- UCI HAR Dataset
- https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset

---

## Results

- Achieved ~95% accuracy using ANN with PCA features
- Improved generalization using Optuna hyperparameter tuning

---

## Author

Built by Abhiyan Paudel
Aspiring AL/ML Engineer

---