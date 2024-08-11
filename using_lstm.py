import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import clone_model

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, **fit_params):
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_config()

    def set_params(self, **params):
        if 'build_fn' in params:
            params.pop('build_fn')
        if 'kwargs' in params:
            params.pop('kwargs')
        self.model = clone_model(self.build_fn(**params))
        return self
# 333333333333333333333333



def download_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    data['Date'] = data.index
    return data

def predict_next_prices_with_dates(last_sequence, last_date, time_steps, interval_hours, num_predictions):
    open_prices = []
    close_prices = []
    high_prices = []
    low_prices = []
    prediction_dates = []

    for _ in range(num_predictions):
        y_pred = model.predict(last_sequence)
        open_pred, high_pred, low_pred, close_pred = y_pred.flatten()
        open_prices.append(open_pred)
        high_prices.append(high_pred)
        low_prices.append(low_pred)
        close_prices.append(close_pred)

        last_sequence = np.roll(last_sequence, shift=-1, axis=1)
        last_sequence[0, -1, :] = [open_pred, high_pred, low_pred, close_pred]

        # Calculate the next date based on the last predicted date and interval
        last_date = last_date + pd.DateOffset(hours=interval_hours)
        prediction_dates.append(last_date)

    return open_prices, high_prices, low_prices, close_prices, prediction_dates


# Streamlit app
st.title("Stock Price Prediction")

# User input for stock symbol, start date, and end date
stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
start_date = st.date_input("Enter start date:", pd.to_datetime("2022-01-01"))
end_date = st.date_input("Enter end date:", pd.to_datetime("2023-01-01"))

# User input for prediction interval and number of predicted points
interval_type = st.selectbox("Select prediction interval type:", ["Hour", "Day", "Minute"])
interval_value = st.number_input("Enter prediction interval value:", min_value=1, value=1)
num_predictions = st.number_input("Enter number of predicted points:", min_value=1, value=10)

# Convert interval type to hours
if interval_type == "Hour":
    interval_hours = interval_value
elif interval_type == "Day":
    interval_hours = interval_value * 24
elif interval_type == "Minute":
    interval_hours = interval_value / 60

# Download stock data
df = download_stock_data(stock_symbol, start_date, end_date)
df = df.dropna()

# Display downloaded stock data
st.subheader(f"Downloaded Stock Data for {stock_symbol}:")
st.write(df)

# Training section
if st.button("Train Model"):
    st.info("Training the model. Please wait...")
    
    # Extract input (X) and output (y) variables
    X = df[['Open', 'High', 'Low', 'Close']].values
    y = df[['Open', 'High', 'Low', 'Close']].values

    # Normalize the data using the scaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    # Prepare the data for LSTM
    time_steps = 24
    X_rnn, y_rnn = [], []

    for i in range(len(X) - time_steps):
        X_rnn.append(X[i:(i + time_steps), :])
        y_rnn.append(y[i + time_steps])

    X_rnn, y_rnn = np.array(X_rnn), np.array(y_rnn)

    # Hyperparameter tuning using GridSearchCV
    def create_model(units=50, activation='relu'):
        model = Sequential()
        model.add(LSTM(units=units, activation=activation, input_shape=(X_rnn.shape[1], X_rnn.shape[2])))
        model.add(Dense(units=4))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

        # ... (existing code)

    # Hyperparameter tuning using GridSearchCV
    from sklearn.model_selection import GridSearchCV

    # Define hyperparameters to search
    param_grid = {'units': [50, 100], 'activation': ['relu', 'tanh']}

    # Create a KerasRegressorWrapper
    keras_wrapper = KerasRegressorWrapper(build_fn=create_model)

    # Create a GridSearchCV object
    grid = GridSearchCV(estimator=keras_wrapper, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_result = grid.fit(X_rnn, y_rnn)

    # Get the best parameters
    best_units = grid_result.best_params_['units']
    best_activation = grid_result.best_params_['activation']

    # Build the final model with the best parameters
    final_model = create_model(units=best_units, activation=best_activation)
    final_model.fit(X_rnn, y_rnn, epochs=20, batch_size=16, verbose=2)  # Adjust epochs and batch_size

    # Store the trained model in st.session_state
    st.session_state.model = final_model
    st.success("Model trained successfully!")

    # ... (continue with the rest of the code)


# Prediction section
if 'model' in st.session_state:
    # Use the trained model to predict the next sequence
    X_pred = X_rnn[-1].reshape((1, time_steps, X_rnn.shape[2]))
    last_date = df['Date'].iloc[-1]
    open_prices, high_prices, low_prices, close_prices, prediction_dates = predict_next_prices_with_dates(
        X_pred, last_date, time_steps, interval_hours, num_predictions
    )

    # Inverse transform predictions
    predicted_prices = scaler.inverse_transform(np.array([open_prices, high_prices, low_prices, close_prices]).T)
    open_prices = predicted_prices[:, 0]
    high_prices = predicted_prices[:, 1]
    low_prices = predicted_prices[:, 2]
    close_prices = predicted_prices[:, 3]

    # Display predicted values with dates
    st.subheader("Predicted Next Prices:")
    predicted_df = pd.DataFrame({
        "Date": prediction_dates,
        "Predicted Open": open_prices.flatten(),
        "Predicted High": high_prices.flatten(),
        "Predicted Low": low_prices.flatten(),
        "Predicted Close": close_prices.flatten()
    })
    st.write(predicted_df)

    # Visualize the actual and predicted sequences separately
    st.subheader("Actual Prices:")
    fig_actual = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Actual Prices')])
    st.plotly_chart(fig_actual)

    st.subheader("Predicted Prices:")
    fig_predicted = go.Figure(data=[go.Candlestick(x=predicted_df['Date'],
                    open=predicted_df['Predicted Open'],
                    high=predicted_df['Predicted High'],
                    low=predicted_df['Predicted Low'],
                    close=predicted_df['Predicted Close'],
                    name='Predicted Prices')])
    st.plotly_chart(fig_predicted)

    st.subheader("Combined Actual and Predicted Prices:")
    fig_combined = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Actual Prices'),
                    go.Candlestick(x=predicted_df['Date'],
                    open=predicted_df['Predicted Open'],
                    high=predicted_df['Predicted High'],
                    low=predicted_df['Predicted Low'],
                    close=predicted_df['Predicted Close'],
                    name='Predicted Prices')])
    st.plotly_chart(fig_combined)

else:
    st.warning("Train the model to make predictions.")
