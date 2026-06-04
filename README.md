# Mental Health Risk Prediction Model

## Overview

This project uses Machine Learning to predict whether an individual is at **High Risk** or **Low Risk** for mental health concerns based on behavioral factors such as age, stress level, and sleep quality.

The model was developed using a public Kaggle dataset and implemented in Python using the K-Nearest Neighbors (KNN) algorithm. The goal was to build an accurate and efficient classification model using simple, non-clinical features that are easy to collect and analyze.

## Dataset

* Source: Kaggle – Mental Health Risk Prediction Dataset
* Records: 1,000 entries
* Features: Age, Sleep Quality, Stress Level, and other behavioral indicators
* Target Variable: Mental Health Risk (High Risk / Low Risk)

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn

## Machine Learning Pipeline

1. Data preprocessing and cleaning
2. Removal of non-informative features
3. Label encoding of target variable
4. Feature scaling using StandardScaler
5. Train-test split (80% training, 20% testing)
6. K-Nearest Neighbors (KNN) model training
7. Performance evaluation and visualization

## Model Configuration

* Algorithm: K-Nearest Neighbors (KNN)
* Neighbors: 7
* Weights: Distance
* Distance Metric: Manhattan

## Results

The tuned KNN model achieved:

* Accuracy: 100%
* Precision: 100%
* Recall: 100%
* F1 Score: 100%

The confusion matrix showed zero misclassifications on the test dataset, demonstrating strong predictive performance.

## Future Improvements

* Evaluate performance on larger and more diverse datasets
* Compare results with Logistic Regression, Random Forest, and Neural Networks
* Incorporate additional behavioral factors such as screen time and physical activity
* Develop a web-based application for real-time mental health risk assessment

## Author

Aieyaan Yeasin

Machine Learning | Data Analysis | Python Development
