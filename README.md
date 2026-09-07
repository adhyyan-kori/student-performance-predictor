# Student Performance Predictor

An ML-powered web application that predicts a student's end-semester performance based on academic, demographic, family, and other student-related information.

Built using Python, Scikit-learn, Streamlit, and SHAP.

## Project Overview

This project uses the UCI Student Academics Performance dataset to train a machine learning classification model that predicts one of four performance categories:

- Best
- Very Good
- Good
- Pass

The trained model is integrated into a Streamlit web application where users can enter student information and receive:

1. A predicted performance category
2. The most influential factor behind the prediction
3. Other factors that influenced the individual prediction
4. The direction of each factor's influence

## Features

- Interactive Streamlit web interface
- Student performance prediction using Random Forest
- Categorical feature encoding using One-Hot Encoding
- Student-specific prediction explanations using SHAP
- Human-readable labels for encoded categories
- Separate input and prediction result pages
- Saved model and encoder for application use

## Machine Learning Workflow

Dataset → Data Loading → Data Cleaning → Exploratory Data Analysis → Train/Test Split → One-Hot Encoding → Model Training → Model Evaluation → Model & Encoder Saved → Streamlit Application → Prediction + SHAP Explanation

## Model

The primary model used in the application is:

**Random Forest Classifier**

The target variable is `esp`, which represents the End Semester Percentage category.

### Model Performance

| Model | Test Accuracy | 5-Fold CV Accuracy |
|---|---:|---:|
| Random Forest | 62.96% | 60.48% |
| Logistic Regression | 70.37% | 60.57% |
| Decision Tree | 48.15% | — |
| SVM | 51.85% | — |
| KNN | 44.44% | — |
| Gradient Boosting | ~59% | — |

Logistic Regression achieved a higher accuracy on the held-out test set, but its cross-validation performance was almost identical to Random Forest.

Random Forest was retained as the application model because it provides a good balance for this project and works naturally with the SHAP tree-based explanation approach used in the application.

> Note: The dataset used in this project is very small (131 records in the downloaded ARFF file). Therefore, model accuracy can vary considerably depending on the train/test split, and the results should not be interpreted as a definitive measure of real-world predictive performance.

## Explainable AI with SHAP

The application uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

Instead of only showing:

**Predicted Performance: Good**

the application also identifies which features had the strongest influence on that particular student's prediction.

A positive SHAP value means that the feature pushed the model's prediction toward the predicted class, while a negative SHAP value means that it pushed the prediction away from the predicted class.

> Note: SHAP explanations describe how features influenced the model's prediction. They do not establish that a feature caused the student's academic outcome.

## Dataset

The project uses the UCI Student Academics Performance Dataset.

The downloaded dataset contains 131 student records and 22 attributes.

The target variable is `esp`, with four performance categories:

- Best
- Very Good (`Vg`)
- Good
- Pass

The project includes attributes related to:

- Gender
- Caste
- 10th grade performance
- 12th grade performance
- Internal assessment
- Arrears
- Living status
- Admission status
- Family income
- Family size
- Parents' qualifications
- Parents' occupations
- Study habits
- School sector
- Medium of instruction
- Travel time
- Attendance

## Technologies Used

- Python — programming language
- Pandas — data manipulation
- NumPy — numerical operations
- Scikit-learn — machine learning
- SciPy — ARFF dataset loading
- Matplotlib — data visualization
- SHAP — model explainability
- Joblib — saving/loading trained models
- Streamlit — web application

## Project Structure

student-performance-predictor/
│
├── app.py
│
├── model_encoder/
│   ├── encoder.pkl
│   └── random_forest_model.pkl
│
├── notebook/
│   └── exploration.ipynb
│
├── student+academics+performance/
│   └── Sapfile1.arff
│
├── .gitignore
└── README.md

## Installation

### 1. Clone the repository

    git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
    cd student-performance-predictor

### 2. Install the required libraries

    pip install pandas scipy scikit-learn matplotlib streamlit shap joblib

### 3. Run the application

    python -m streamlit run app.py

The Streamlit application will open in your browser.

## Project Goal

The goal of this project was to build an end-to-end machine learning application rather than only training a model in a notebook.

It demonstrates the complete process of:

- Working with a real-world dataset
- Performing exploratory data analysis
- Preparing categorical data
- Training and comparing classification algorithms
- Evaluating model performance
- Saving a trained model
- Integrating the model into a web application
- Providing interpretable, student-specific predictions

## Limitations

This project is intended as an educational/academic machine learning project.

The dataset is relatively small, so the model's performance may not generalize well to other student populations.

Additionally, SHAP explanations indicate how features influenced the model's prediction. They do not establish that those features actually caused a student's academic outcome.

## Dataset Source

UCI Student Academics Performance Dataset

Dataset provided by the UCI Machine Learning Repository.

## Author

Adhyyan Kori

Built as a machine learning and Streamlit project.