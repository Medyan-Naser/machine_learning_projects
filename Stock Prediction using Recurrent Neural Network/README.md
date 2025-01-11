# Stock Price Prediction Using Recurrent Neural Network (RNN)

This project demonstrates how to predict stock prices using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The model is trained on historical stock data and used to predict the future prices of a specific stock.

## Overview

We use an RNN with LSTM layers to model the sequential nature of stock price data. The model is trained on historical stock prices, with data preprocessing steps such as feature scaling and the creation of time windows. The network architecture incorporates multiple LSTM layers to capture temporal dependencies, alongside Dropout layers to mitigate overfitting.

## Data

The model is trained on historical stock prices, which include the following:
- **Training Data**: Historical stock prices used for training the model.
- **Test Data**: Recent stock prices used to test and validate the model's predictions.

## Preprocessing

The preprocessing steps involve:
1. **Feature Scaling**: The stock prices are normalized to a range between 0 and 1 using MinMaxScaler to enhance the model's performance.
2. **Creating Data Structure**: A rolling window of 60 timesteps is used to generate input features that predict the next stock price (output). This is applied to both the training and test sets.

## Architecture

### 1. LSTM Layer (First)
- **Explanation**: The first LSTM layer captures the sequential dependencies in the time-series data. With the parameter `return_sequences=True`, it outputs the full sequence of hidden states, which is passed to the next LSTM layer.
- **Purpose**: This layer is crucial for learning the temporal dependencies within the stock price data. Its ability to capture the relationships between previous stock prices enables the model to make informed predictions about future prices.

### 2. Dropout (After First LSTM)
- **Explanation**: Dropout regularization is applied to the output of the first LSTM layer, randomly setting a fraction of units to zero during training.
- **Purpose**: The inclusion of Dropout helps prevent overfitting by encouraging the model to generalize well across different subsets of the data, thus improving its robustness and performance on unseen data.

### 3. LSTM Layer (Second)
- **Explanation**: A second LSTM layer is added to further model the complex temporal dependencies in the data. This layer also uses `return_sequences=True` to pass the sequence of hidden states to the next layer.
- **Purpose**: This additional LSTM layer allows the network to learn more intricate patterns from the data, improving the model's ability to capture long-range dependencies and making it more effective in predicting stock prices.

### 4. Dropout (After Second LSTM)
- **Explanation**: Another Dropout regularization layer is added after the second LSTM layer.
- **Purpose**: By applying Dropout once again, the model is further regularized, reducing the risk of overfitting and encouraging it to learn more generalized features rather than memorizing the training data.

### 5. LSTM Layer (Third)
- **Explanation**: A third LSTM layer is added to model even more complex patterns within the time-series data.
- **Purpose**: The third LSTM layer is critical for capturing deeper and more abstract temporal relationships in the stock price data, which enhances the model’s predictive capabilities.

### 6. Dropout (After Third LSTM)
- **Explanation**: Dropout regularization is applied after the third LSTM layer to continue preventing overfitting.
- **Purpose**: This layer further helps the model to generalize, ensuring it does not overfit to specific patterns in the training data, thus improving its ability to make accurate predictions on new, unseen data.

### 7. LSTM Layer (Fourth)
- **Explanation**: The fourth LSTM layer is added without the `return_sequences` parameter, as it is the final layer before the output layer.
- **Purpose**: This layer consolidates the temporal information learned by the previous LSTM layers and prepares the data for the output layer, ensuring that the model is ready to make a prediction based on all the learned patterns.

### 8. Dropout (After Fourth LSTM)
- **Explanation**: Dropout regularization is applied after the fourth LSTM layer.
- **Purpose**: This final Dropout layer continues to reduce the likelihood of overfitting and ensures that the model remains robust by focusing on diverse features learned across all layers.

### 9. Dense Layer (Output Layer)
- **Explanation**: The Dense layer outputs a single value, which is the predicted stock price for the next time step.
- **Purpose**: As the final layer in the network, the Dense layer produces the model’s prediction. This layer takes the aggregated temporal information from the previous LSTM layers and generates a single output: the predicted future stock price.

## Training

The model is trained using the **Adam** optimizer and the **mean squared error** loss function. It undergoes 100 epochs with a batch size of 32. The training loss is monitored and plotted to track the model’s learning process and to evaluate the effectiveness of the optimization.

## Results

Once trained, the model is used to make predictions on the test dataset. The real stock prices and the predicted stock prices are visualized to assess the model’s performance and to understand how well it can forecast future prices.
- **Stock Price Prediction Plot**: Compares the predicted stock prices with the actual stock prices, highlighting the model’s accuracy in forecasting.
![RNN_stock_price_prediction](https://github.com/user-attachments/assets/b3d417e9-f6e5-4538-b135-bbf44ae614aa)


## Conclusion

Conclusion

The RNN with LSTM architecture is well-suited for stock price prediction, as it effectively captures the temporal dependencies inherent in time-series data. The model, with its multiple LSTM layers and regularization techniques like Dropout, demonstrates a robust ability to forecast stock prices. Its performance could potentially be further enhanced by incorporating more advanced techniques or additional features such as market indicators.
