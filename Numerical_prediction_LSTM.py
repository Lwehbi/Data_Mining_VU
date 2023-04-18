import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

df = pd.read_csv(r'C:\Users\Laith Wehbi\OneDrive - Vrije Universiteit Amsterdam\Desktop\preprocessed_dataset_nofeatures.csv')

df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y')
df = df.sort_values('time')

# Set the index to 'time' for time-series modeling
df.set_index('time', inplace=True)

# Drop unnecessary columns
df.drop(['id', 'label', 'day_id'], axis=1, inplace=True)

# Train-test split (80/20)
train_size = int(len(df) * 0.8)
train, test = df.values[:train_size, :], df.values[train_size:, :]

# Create input-output pairs for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Create LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=2)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate mse and mae
mse = mean_squared_error(test[:-1, 0], test_predictions[:, 0])
mae = mean_absolute_error(test[:-1, 0], test_predictions[:, 0])
print(f'MSE: {mse}')
print(f'MAE: {mae}')

# Print the model's architecture
model.summary()

# Calculate the train and test index ranges
train_range = range(len(train) - 1)
test_range = range(len(train) - 1, len(train) + len(test) - 2)

# Plot the actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_range, test[:-1, 0], label='Real values')
plt.plot(test_range, test_predictions[:, 0], label='Predicted values')
plt.xlabel('Index')
plt.ylabel('Mood')
plt.legend()
plt.show()

