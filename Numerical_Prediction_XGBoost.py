import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


df = pd.read_csv(r'C:\Users\Laith Wehbi\OneDrive - Vrije Universiteit Amsterdam\Desktop\preprocessed_dataset_nofeatures.csv')

# Create some lagged features for mood (this is typically done with xgboost)
df['mood_lag1'] = df['mood'].shift(1)
df['mood_lag2'] = df['mood'].shift(2)
df['mood_lag3'] = df['mood'].shift(3)
df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y')
df['time'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Drop instances (rows) with missing values (this is done just as a precaution)
df.dropna(inplace=True)

# Split the data into features and target
X = df.drop(['id', 'mood', 'label', 'day_id'], axis=1)
y = df['mood']

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialise XGBoost
base_model = xgb.XGBRegressor()

# Set up parameter grid to use for grid search
param_grid = {
    'objective': ['reg:squarederror'],
    'max_depth': [3, 5, 7, 10],
    'eta': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300]
}

# Perform 1) grid search with cross-validation to find the best parameters and 2) then get the best parameters from the grid search
grid_search = GridSearchCV(base_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the XGBoost model with the best parameters and make predictions
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Calculate mse and mae
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'MSE: {mse}')
print(f'MAE: {mae}')

# Plot the real mood values versus the predicted ones over time
fig, ax = plt.subplots()
ax.scatter(pd.to_datetime(X_test['time'], unit='s'), y_test, label='Actual', marker='.')
ax.scatter(pd.to_datetime(X_test['time'], unit='s'), y_pred, label='Predicted', marker='.')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xlabel('Time (MM-DD)')
plt.ylabel('Mood')
plt.legend()
plt.show()

# Get feature importance
importances = best_model.feature_importances_
# Create and sort new feature importance df
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feature_importances.sort_values(by="Importance", ascending=False, inplace=True)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_importances["Feature"], feature_importances["Importance"])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.xticks(rotation=90)
plt.show()