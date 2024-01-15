# Bitcoin Price Prediction Using LSTM

## Objective
The purpose of this project was to develop a deep learning model to predict the price of Bitcoin. Specifically, an LSTM model was developed to forecast future prices based on historical data.

## Data Source
The dataset contains hourly Bitcoin price data including open, high, low, close, and volume. It was sourced from [Kaggle]("https://www.kaggle.com/datasets/olegshpagin/crypto-coins-prices-ohlcv").

## Data Preprocessing and Feature Engineering
The preprocessing steps included:
- Cleaning the data to handle any missing values.
- Formatting the datetime column.
- Generating lag features to capture temporal dependencies.
- Calculating technical indicators such as RSI, MACD, and others.
- Normalizing features using `MinMaxScaler` to ensure the model receives input data within a scaled range.

Feature engineering involved:
- Creating lagged versions of price features to provide past context.
- Adding day-of-week information to account for any seasonality.

## Model Architecture
The LSTM model consists of:
- An input layer shaped to the preprocessed data.
- LSTM layers to capture long-term dependencies within the time series data.
- Dropout layers to mitigate overfitting.
- A dense output layer to predict continuous price values.

An LSTM model was used for its ability to remember information over long periods, which is important for time series forecasting.

## Evaluation Metrics
The model was evaluated using the following metrics:
- Mean Squared Error (MSE) for loss calculation.
- Root Mean Squared Error (RMSE) to assess the average error magnitude.
- Mean Absolute Error (MAE) to understand prediction accuracy.

Results were interpreted in the context of actual price scales to provide clear understanding of the model's predictive ability.

## Model Deployment

This project does not include a live deployment of the model. However, if one were to deploy the LSTM model for real-time Bitcoin price prediction, the following steps could be considered:

### Deployment Strategy
1. **Cloud Hosting:** The model could be hosted on a cloud platform like AWS, Azure, or Google Cloud Platform.
2. **Containerization:** The model and its dependencies can be containerized using Docker, which can then be deployed to a Kubernetes cluster.
3. **Serverless Functions:** Platforms like AWS Lambda or Google Cloud Functions could be used to deploy the model as a serverless function.

### Real-Time Data Fetching
- An API could be set up to fetch real-time Bitcoin price data from crypto exchanges or financial data providers.
- This data could then be preprocessed and normalized in the same way as the training data before being fed into the model for prediction.

### Making Live Predictions
- The deployed model could be exposed via a REST API endpoint, allowing users or applications to make HTTP requests to get live predictions.
- For a trading system, these predictions could be used to trigger buy or sell orders based on certain thresholds or trading strategies.

## Results and Limitations

### Results
The LSTM model demonstrated promising results. With a Mean Absolute Error (MAE) of approximately 1.57% of the mean actual price and a Root Mean Squared Error (RMSE) of nearly 1.99%, the model is able to predict Bitcoin prices with reasonable accuracy. These metrics suggest that the model captures the general trends in the price of Bitcoin pretty well. However, it's still important to consider these results in the context of market volatility and potential overfitting.

### Limitations
While the model's performance is encouraging, there are several limitations to consider:
- **Market Volatility:** Bitcoin is highly volatile, and the model may not account for sudden market changes due to its reliance on historical data.
- **Overfitting:** Although early stopping is used during training to mitigate overfitting, the possibility of the model being too closely fitted to the training data cannot be ignored without further validation.
- **Hyperparameter Selection:** The choice of hyperparameters was based on common practices and some experimentation. Further tuning and validation are necessary to optimize the model's performance.
- **Data Diversity:** The model was trained on historical hourly price data. Incorporating a more diverse set of features, including macroeconomic indicators or blockchain activity data, could potentially improve the model's predictions.
- **Feature Engineering:** The process of creating lag features, RSI, and MACD indicators, although standard in its approach, may not capture all predictive signals. Advanced feature engineering could yield better inputs for the model.

### Improvements and Extensions
To improve the model's accuracy, the following steps could be taken:
- **Model Ensembling:** Combining the LSTM with other models or using ensemble methods could help improve predictions by capturing a wider array of patterns.
- **Advanced Hyperparameter Tuning:** Using techniques like grid search or Bayesian optimization to find the optimal set of hyperparameters.
- **Regularization Techniques:** Using regularization methods such as L1 or L2 regularization, or dropout layers in the neural network architecture to reduce overfitting.
- **Cross-Validation:** Using time-series cross-validation to assess the model's performance across different time periods.
- **Inclusion of Exogenous Variables:** Adding external factors that affect Bitcoin prices, such as sentiment analysis from social media or news, could provide additional predictive ability.

## Running the Project
To run this project, you will need to:
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Execute the source code in the correct order as labeled.
4. The model and scaler are saved and can be loaded for evaluation or further training.

## Dependencies
The project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `scikit-learn`
- `jupyter`

These can be installed by running `pip install -r requirements.txt` where `requirements.txt` contains all the necessary packages.

For any further questions or contributions to the project, please submit an issue or pull request.
