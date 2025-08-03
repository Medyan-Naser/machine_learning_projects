
## Comparing different regression types

```python

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)

```

## Associated functions commonly used

Splits the dataset into training and testing subsets to evaluate the model's performance.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

Standardizes features by removing the mean and scaling to unit variance.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.metrics import log_loss
loss = log_loss(y_true, y_pred_proba)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)

from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))


```

Computes the R-squared value, indicating how well the model explains the variability of the target variable.

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```


Transforms categorical features into a one-hot encoded matrix

```python

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(categorical_data)

```

Computes the accuracy of a classifier by comparing predicted and true labels.

```python

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)

```


Encodes labels (target variable) into numeric format.

```python

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

```


Plots a decision tree model for visualization.

```python
from sklearn.tree import plot_tree
plot_tree(model, max_depth=3, filled=True)
```

Scales each feature to have zero mean and unit variance (standardization).

```python
from sklearn.preprocessing import normalize
normalized_data = normalize(data, norm='l2')
```

Computes sample weights for imbalanced datasets.

```python
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight(class_weight='balanced', y=y)
```

Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for binary classification models.

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_score)
```

```python



```

```python



```