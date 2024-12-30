import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

## Simple Linear Regression
# Get the directory of the currently running script
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'Salary_Data.csv')
# Importing the dataset
dataset = pd.read_csv(file_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results￼
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.suptitle('Simple Linear Regression', fontsize=14)
plt.show()


## Multiple Linear Regression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Get the directory of the currently running script
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, '50_Startups.csv')

# Importing the dataset
dataset = pd.read_csv(file_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Label Encoding (if needed)
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

# OneHot Encoding using ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [3])  # Apply OneHotEncoder to column 3
    ],
    remainder='passthrough'  # Keep other columns unchanged
)
X = column_transformer.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Profit')
plt.legend()
plt.grid(True)
plt.suptitle('Multiple Linear Regression', fontsize=14)
plt.show()
