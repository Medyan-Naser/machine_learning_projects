# Fraud Detection using Artificial Neural Networks

This project utilizes an Artificial Neural Network (ANN) as well as Self-Organizing Maps (SOM) to detect fraudulent activity in customer transaction data. By identifying patterns of normal behavior and detecting anomalies, these models help to pinpoint potentially fraudulent transactions.

## Dataset

The dataset consists of customer transaction data, which includes the following features:

- **CreditScore**: The credit score of the customer.
- **Geography**: The country of residence.
- **Gender**: The gender of the customer.
- **Age**: The age of the customer.
- **Tenure**: How long the customer has been with the bank.
- **Balance**: The account balance of the customer.
- **NumberOfProducts**: The number of products the customer has.
- **HasCrCard**: Whether the customer has a credit card (binary).
- **IsActiveMember**: Whether the customer is an active member (binary).
- **EstimatedSalary**: The estimated salary of the customer.

The target variable is whether the customer is likely to churn (fraudulent activity indicator).

## Objective

The goal of this project is to build models that can predict if a customer is at risk of committing fraudulent activity using both neural networks and unsupervised SOM clustering. The models leverage transaction data to detect anomalies and suspicious behavior.

## Data Preprocessing

Before training the model, the following preprocessing steps were applied:

1. **Feature Selection**: Relevant features such as demographic and transactional data were selected for analysis.
2. **Encoding Categorical Variables**: The 'Gender' column was encoded using label encoding, and 'Geography' was One-Hot Encoded to avoid the dummy variable trap.
3. **Feature Scaling**: All features were scaled using StandardScaler to normalize the data for optimal performance of the neural network.

## Building the Artificial Neural Network (ANN)

### Layers Overview

1. **Input Layer (First Hidden Layer)**: 
    - **Units**: 6
    - **Activation**: ReLU (Rectified Linear Unit)
    - **Purpose**: The input layer receives the feature values and starts the process of extracting patterns from the data. The ReLU activation function introduces non-linearity to the model and helps it learn complex patterns in the data.

2. **Second Hidden Layer**:
    - **Units**: 6
    - **Activation**: ReLU
    - **Purpose**: The second hidden layer helps the model learn deeper patterns from the data, increasing the ability to classify complex scenarios. The ReLU activation function ensures efficient learning by providing a sparsity mechanism.

3. **Output Layer**:
    - **Units**: 1
    - **Activation**: Sigmoid
    - **Purpose**: The output layer predicts a binary outcome, where a value closer to 1 indicates fraudulent activity and a value closer to 0 indicates a non-fraudulent transaction. The sigmoid activation function outputs probabilities, which are suitable for binary classification tasks.

### Model Compilation and Training

- **Optimizer**: Adam, an adaptive learning rate optimization algorithm, is used to efficiently minimize the loss.
- **Loss Function**: Binary Cross-Entropy, which is ideal for binary classification problems.
- **Metrics**: Accuracy, to evaluate the proportion of correct predictions.

The model was trained for 100 epochs with a batch size of 32, and training data was split with 10% used for validation to assess performance.

## Self-Organizing Maps (SOM) for Fraud Detection

In addition to using Artificial Neural Networks, a Self-Organizing Map (SOM) was applied to detect fraudulent activity in customer data. SOM is an unsupervised learning algorithm that uses clustering to find patterns and anomalies in data.

The SOM method was applied to the dataset by clustering similar data points and identifying any outliers that could be indicative of fraudulent activity. Fraudulent customers were identified by examining the neurons in the SOM grid that showed a high distance from other neurons, which correspond to anomalous behavior.


## Conclusion

This project successfully built a neural network and applied Self-Organizing Maps (SOM) to identify fraudulent activity in customer transaction data. Both methods offer valuable insights and can be used together to improve the accuracy and robustness of fraud detection systems in financial institutions.


