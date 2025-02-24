# Cryptocurrency Price Prediction using LSTM

## Introduction
This project predicts the next-day closing price of Enjin Coin (ENJ) using a deep learning model based on Long Short-Term Memory (LSTM) networks. The dataset contains historical cryptocurrency data, including the date, opening price, closing price, highest and lowest prices, volume, and other relevant financial metrics.

## Dataset
- The dataset is downloaded from Kaggle using the `kagglehub` library.
- The specific file used: **Enjin Coin.csv**
- Data contains multiple cryptocurrencies, but only Enjin Coin's data is used.
- Columns include:
  - Date
  - Open price
  - High price
  - Low price
  - Close price
  - Volume
  - Market Cap

## Data Preprocessing
1. **Load Data**: Read the CSV file into a pandas DataFrame.
2. **Convert Date Format**: Convert the `Date` column to datetime format and sort the data in chronological order.
3. **Check for Missing Values**: Identify and handle any missing values by filling or removing them.
4. **Feature Engineering**:
   - Extract `day`, `month`, and `year` from the `Date` column.
   - Create additional time-based features such as moving averages.
5. **Drop Unnecessary Columns**: Remove the `Currency` column and other irrelevant fields.
6. **Scale Data**: Normalize the `Close` price and other numerical features using `MinMaxScaler`.

## Model Training
1. **Create Sequences**:
   - Use a sliding window technique to generate sequences of 60 past closing prices as input and the next-day closing price as the target.
   - Ensure sequences are properly reshaped for LSTM input.
2. **Split Data**:
   - Divide the dataset into training (80%) and testing (20%) sets.
   - Reshape data into three dimensions required for LSTM models.
3. **Build the LSTM Model**:
   - Input Layer: LSTM layer with 50 units (return sequences enabled).
   - Second LSTM Layer: 50 units (return sequences disabled).
   - Dense Layers: Two fully connected layers (25 units and 1 unit output layer).
4. **Compile & Train**:
   - Optimizer: Adam
   - Loss Function: Mean Squared Error (MSE)
   - Batch Size: 32
   - Epochs: 10
   - Monitor validation loss to prevent overfitting.

## Model Evaluation & Prediction
- Predictions are generated on the test set.
- The predictions are inverse-transformed to get the actual price values.
- Model performance is evaluated using:
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
  - **R-squared Score (R²)**

## Results & Next Steps
- The model successfully learns patterns in historical price data.
- RMSE provides insight into how accurate the predictions are.
- Observations:
  - Some underfitting or overfitting may occur depending on hyperparameters.
  - Model predictions may lag behind real trends.
- Future improvements:
  - Increase training epochs.
  - Tune hyperparameters.
  - Incorporate additional features such as trading volume and market sentiment.
  - Experiment with different deep learning architectures like GRU or Transformer models.

## Tools Used
- **Python Libraries**: Pandas, NumPy, Seaborn, TensorFlow/Keras, Scikit-learn, Matplotlib
- **Machine Learning Algorithm**: LSTM (Long Short-Term Memory)
- **Dataset Source**: Kaggle
- **Development Environment**: Jupyter Notebook, Google Colab

## Additional Considerations
- The model does not account for external market events, news, or investor sentiment, which can influence cryptocurrency prices significantly.
- Real-time implementation would require fetching live price data and retraining periodically.
- Deployment could be done using Flask or FastAPI to create a web-based prediction service.

