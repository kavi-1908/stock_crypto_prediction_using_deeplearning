# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import SimpleRNN, Dense
# import yfinance as yf

# def download_stock_data(symbol, start_date, end_date):
#     data = yf.download(symbol, start=start_date, end=end_date, progress=False)
#     data['Date'] = data.index
#     return data

# def predict_next_points_with_dates(last_sequence, last_date, time_steps, interval_hours):
#     predictions = []
#     prediction_dates = []

#     for _ in range(40):
#         y_pred = model.predict(last_sequence)
#         predictions.append(y_pred.flatten()[0])
#         last_sequence = np.roll(last_sequence, shift=-1, axis=1)
#         last_sequence[0, -1, 0] = y_pred.flatten()[0]

#         # Calculate the next date based on the last predicted date and interval
#         last_date = last_date + pd.DateOffset(hours=interval_hours)
#         prediction_dates.append(last_date)

#     return predictions, prediction_dates

# # Streamlit app
# st.title("Stock Price Prediction")

# # User input for stock symbol, start date, and end date
# stock_symbol = st.text_input("Enter stock symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
# start_date = st.date_input("Enter start date:", pd.to_datetime("2022-01-01"))
# end_date = st.date_input("Enter end date:", pd.to_datetime("2023-01-01"))

# # Download stock data
# df = download_stock_data(stock_symbol, start_date, end_date)
# df = df.dropna()

# # Display downloaded stock data
# st.subheader(f"Downloaded Stock Data for {stock_symbol}:")
# st.write(df)

# # Training section
# if st.button("Train Model"):
#     st.info("Training the model. Please wait...")
    
#     # Extract input (X) and output (y) variables
#     X = df[['Open', 'High', 'Low']].values
#     y = df['Close'].values

#     # Normalize the data using the scaler
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#     y = scaler.fit_transform(y.reshape(-1, 1))

#     # Prepare the data for SimpleRNN
#     time_steps = 24
#     X_rnn, y_rnn = [], []

#     for i in range(len(X) - time_steps):
#         X_rnn.append(X[i:(i + time_steps), :])
#         y_rnn.append(y[i + time_steps])

#     X_rnn, y_rnn = np.array(X_rnn), np.array(y_rnn)

#     # Build and train SimpleRNN model
#     model = Sequential()
#     model.add(SimpleRNN(units=50, activation='relu', input_shape=(X_rnn.shape[1], X_rnn.shape[2])))
#     model.add(Dense(units=50, activation='relu'))
#     model.add(Dense(units=50, activation='relu'))
#     model.add(Dense(units=1))
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     model.fit(X_rnn, y_rnn, epochs=10, batch_size=1, verbose=2)

#     # Store the trained model in st.session_state
#     st.session_state.model = model
#     st.success("Model trained successfully!")

# # Prediction section
# if 'model' in st.session_state:
#     # Use the trained model to predict the next sequence
#     X_pred = X_rnn[-1].reshape((1, time_steps, X_rnn.shape[2]))
#     last_date = df['Date'].iloc[-1]
#     predictions, prediction_dates = predict_next_points_with_dates(X_pred, last_date, time_steps, interval_hours=1)

#     # Inverse transform predictions
#     predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

#     # Display predicted values with dates
#     st.subheader("Predicted Next 10 Close Prices:")
#     predicted_df = pd.DataFrame({"Date": prediction_dates, "Predicted Close": predictions.flatten()})
#     st.write(predicted_df)

#     # Visualize the predicted sequence along with the actual data
#     st.line_chart(df.set_index('Date')['Close'].rename("Actual Close Price"))
#     st.line_chart(predicted_df.set_index('Date'))
#     st.line_chart(pd.concat([df.set_index('Date')['Close'].rename("Actual Close Price"), predicted_df.set_index('Date')['Predicted Close']], axis=1))
# else:
#     st.warning("Train the model to make predictions.")
# List of Indian stock symbols
# indian_stocks = [
#     "TCS.NS","IRFC.NS","RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
#     "HINDUNILVR.NS", "KOTAKBANK.NS", "ITC.NS", "LT.NS", "HDFC.NS",
#     "BAJAJFINSV.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "SBIN.NS",
#     "M&M.NS", "WIPRO.NS", "ASIANPAINT.NS", "ONGC.NS", "NESTLEIND.NS",
#     "ULTRACEMCO.NS", "SUNPHARMA.NS", "IOC.NS", "MARUTI.NS", "HCLTECH.NS",
#     "TECHM.NS", "POWERGRID.NS", "NTPC.NS", "RECLTD.NS", "COALINDIA.NS",
#     "HINDALCO.NS", "CIPLA.NS", "DRREDDY.NS", "ADANIPORTS.NS", "UPL.NS",
#     "GAIL.NS", "INFIBEAM.NS", "DABUR.NS", "PAGEIND.NS", "PNB.NS",
#     "ZEEL.NS", "VEDL.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "LUPIN.NS"
# ]
# crypto_symbols = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 'ADA-USD', 'DOT1-USD', 'XLM-USD',
#                   'USDT-USD', 'LINK-USD', 'DOGE-USD', 'UNI3-USD', 'BNB-USD', 'WBTC-USD', 'USDC-USD', 'EOS-USD',
#                   'TRX-USD', 'AAVE-USD', 'XTZ-USD', 'SOL1-USD']

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
import yfinance as yf

def download_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    data['Date'] = data.index
    return data

def predict_next_points_with_dates(last_sequence, last_date, time_steps, interval_hours, num_predictions):
    predictions = []
    prediction_dates = []

    for _ in range(num_predictions):
        y_pred = model.predict(last_sequence)
        predictions.append(y_pred.flatten()[0])
        last_sequence = np.roll(last_sequence, shift=-1, axis=1)
        last_sequence[0, -1, 0] = y_pred.flatten()[0]

        # Calculate the next date based on the last predicted date and interval
        last_date = last_date + pd.DateOffset(hours=interval_hours)
        prediction_dates.append(last_date)

    return predictions, prediction_dates

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
    X = df[['Open', 'High', 'Low']].values
    y = df['Close'].values

    # Normalize the data using the scaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))

    # Prepare the data for SimpleRNN
    time_steps = 24
    X_rnn, y_rnn = [], []

    for i in range(len(X) - time_steps):
        X_rnn.append(X[i:(i + time_steps), :])
        y_rnn.append(y[i + time_steps])

    X_rnn, y_rnn = np.array(X_rnn), np.array(y_rnn)

    # Build and train SimpleRNN model
    model = Sequential()
    model.add(SimpleRNN(units=50, activation='relu', input_shape=(X_rnn.shape[1], X_rnn.shape[2])))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
   
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_rnn, y_rnn, epochs=10, batch_size=1, verbose=2)

    # Store the trained model in st.session_state
    st.session_state.model = model
    st.success("Model trained successfully!")

# Prediction section
if 'model' in st.session_state:
    # Use the trained model to predict the next sequence
    X_pred = X_rnn[-1].reshape((1, time_steps, X_rnn.shape[2]))
    last_date = df['Date'].iloc[-1]
    predictions, prediction_dates = predict_next_points_with_dates(X_pred, last_date, time_steps, interval_hours, num_predictions)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Display predicted values with dates
    st.subheader("Predicted Next Close Prices:")
    predicted_df = pd.DataFrame({"Date": prediction_dates, "Predicted Close": predictions.flatten()})
    st.write(predicted_df)

    # Visualize the predicted sequence along with the actual data
    st.line_chart(df.set_index('Date')['Close'].rename("Actual Close Price"))
    st.line_chart(predicted_df.set_index('Date'))
    st.line_chart(pd.concat([df.set_index('Date')['Close'].rename("Actual Close Price"), predicted_df.set_index('Date')['Predicted Close']], axis=1))
else:
    st.warning("Train the model to make predictions.")
