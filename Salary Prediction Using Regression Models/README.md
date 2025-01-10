# Regression Models for Predictive Analysis

In this project, we explore the performance of five regression models: Linear Regression, Multiple Linear Regression, Polynomial Regression, Decision Tree Regression, Random Forest Regression, and Support Vector Regression (SVR). The goal is to apply these models to datasets with varying complexity and compare their performance, particularly focusing on prediction accuracy.

## Objective

The objective of this project is to evaluate and compare the performance of five regression models:

## Models Implemented

The following regression algorithms are utilized:

1. **Linear Regression**: A linear model for predicting a continuous target variable based on a linear relationship with one independent variable.

2. **Multiple Linear Regression**: An extension of Linear Regression, using multiple independent variables to model the relationship with the target variable.

3. **Polynomial Regression**: A model that includes polynomial terms to capture non-linear relationships in the data.

4. **Decision Tree Regression**: A non-linear model that splits data into subsets based on decision rules to predict the target variable.

5. **Random Forest Regression**: An ensemble method that averages predictions from multiple decision trees to improve accuracy and reduce overfitting.

6. **Support Vector Regression (SVR)**: A model that fits a line within a predefined margin while minimizing prediction errors, suitable for non-linear relationships.

## Models Applied

Each model was tested on the same dataset, allowing for a direct comparison of their performance. The models were trained using the training set and tested on the test set. Performance metrics such as Mean Squared Error (MSE) and R-squared were used to evaluate how well each model made predictions.

## Results

- **Linear and Multiple Linear Regression**: Best suited for datasets with linear relationships.
- **Polynomial Regression**: Performs better when dealing with non-linear relationships in data.
- **Decision Tree Regression**: Works well for non-linear datasets but is prone to overfitting without proper tuning.
- **Random Forest Regression**: Provides the most accurate predictions by combining multiple decision trees, handling overfitting effectively.
- **Support Vector Regression (SVR)**: Performs well with non-linear relationships and datasets that have high-dimensional features.

## Visualizations

<div align="center">
    <img src="https://github.com/user-attachments/assets/cf276372-3e8c-4cdd-a618-f54ad968c339" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/2007ee6d-bfc8-4a7f-95d6-9b9610391a79" alt="SVM" width="45%" style="margin: 0 2%;">
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/d19d8475-e514-4748-a22a-54bac636c195" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/b1a01e79-89cb-4f18-9758-fd58f46675bb" alt="SVM" width="45%" style="margin: 0 2%;">
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/f65993a9-a3f1-489c-946a-cdaefe8adff7" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/5b53577d-4ab3-4ea4-959e-e34634e3fbb0" alt="SVM" width="45%" style="margin: 0 2%;">
</div>


## Conclusion

This project demonstrates how different regression models can be applied to datasets with varying complexities. By comparing their results, we gain insights into the most effective models for different types of data relationships. Random Forest Regression showed the best overall performance, with Support Vector Regression also performing well in high-dimensional and non-linear scenarios.
