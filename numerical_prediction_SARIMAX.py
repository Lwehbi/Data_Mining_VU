import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(r'C:\Users\Laith Wehbi\OneDrive - Vrije Universiteit Amsterdam\Desktop\preprocessed_dataset_nofeatures.csv')

df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y')
df['time'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# 80/20 train-test split
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Find the best SARIMAX model using auto_arima
best_model = auto_arima(train_df['mood'], exog=train_df.drop(['id', 'mood', 'label', 'day_id'], axis=1),
                        seasonal=False, stepwise=True, suppress_warnings=True,
                        maxiter=200, trace=True)

print(best_model.summary())

# Fit the best SARIMAX model
results = best_model.fit(train_df['mood'], exog=train_df.drop(['id', 'mood', 'label', 'day_id'], axis=1))

# Make predictions on the test set
y_pred = results.predict(n_periods=len(test_df), exog=test_df.drop(['id', 'mood', 'label', 'day_id'], axis=1))

# Calculate mse and mae
mse = mean_squared_error(test_df['mood'], y_pred)
mae = mean_absolute_error(test_df['mood'], y_pred)
print(f'MSE: {mse}')
print(f'MAE: {mae}')

# Plot the real mood values versus the predicted ones over time
fig, ax = plt.subplots()
ax.scatter(pd.to_datetime(test_df['time'], unit='s'), test_df['mood'], label='Actual', marker='.')
ax.scatter(pd.to_datetime(test_df['time'], unit='s'), y_pred, label='Predicted', marker='.')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xlabel('Time (MM-DD)')
plt.ylabel('Mood')
plt.legend()
plt.show()
