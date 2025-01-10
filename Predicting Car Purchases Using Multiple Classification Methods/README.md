# Customer Purchase Prediction Using Classification Models

This project applies various classification algorithms to predict whether a customer will purchase a car based on demographic and financial attributes.

## Dataset

The dataset includes the following features:

- **User ID**: Unique identifier for each user (not used for prediction).
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Estimated Salary**: Approximate annual salary of the customer.
- **Purchased**: Target variable indicating whether the customer purchased the car (1 = Yes, 0 = No).

## Data Preprocessing

To prepare the data for modeling, the following preprocessing steps are applied:

1. **Encoding Categorical Variables**: Converting 'Gender' into numerical values.
2. **Feature Scaling**: Standardizing 'Age' and 'Estimated Salary' to ensure equal importance.
3. **Splitting the Dataset**: Dividing the data into training and testing sets to evaluate model performance.

## Models Implemented

The following classification algorithms are utilized:

1. **Logistic Regression**: A linear model for binary classification.
2. **K-Nearest Neighbors (KNN)**: A non-parametric method that classifies based on the majority class among the k-nearest neighbors.
3. **Support Vector Machine (SVM)**: A model that finds the optimal hyperplane separating the classes.
4. **Kernel SVM**: An extension of SVM that handles non-linear data using kernel functions.
5. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
6. **Decision Tree Classification**: A model that splits the data into subsets based on feature values.
7. **Random Forest Classification**: An ensemble method that combines multiple decision trees to improve performance.

## Model Evaluation

Each model's performance is evaluated using the following metrics:

- **Confusion Matrix**: To assess true positives, true negatives, false positives, and false negatives.
- **Accuracy Score**: To determine the proportion of correctly predicted instances.


<div align="center">
    <img src="https://github.com/user-attachments/assets/80f2f5cb-73ca-49c2-a087-517c717515f6" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/465e56c7-325a-4781-ac9b-77ff2c890eaf" alt="SVM" width="45%" style="margin: 0 2%;">
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/9d28c95c-f6e6-4fc6-9142-7b911b0614fa" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/fc431758-8210-4895-bfb2-5b544be793e5" alt="SVM" width="45%" style="margin: 0 2%;">
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/e7677171-187d-400f-8df9-4cf802e838b5" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/7e0c0c58-a48a-4159-b661-76197f8dc358" alt="SVM" width="45%" style="margin: 0 2%;">
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/2c9ef93c-78bb-4641-976d-09346707232e" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
</div>


## Conclusion

This project demonstrates the application of various classification algorithms to predict customer purchase behavior. The results indicate that Random Forest performed the best, achieving an accuracy of 89%.
