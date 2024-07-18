import pandas as pd
from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
# Load the dataset, skipping the first 27 rows and taking only the first column of actual data
df = pd.read_excel('consumption_allyear.xlsx', skiprows=27, usecols=[0, 1])
df.columns = ['Date', 'Electricity_Consumption']
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Split the data into train and test sets
train, test = train_test_split(df, train_size=0.8)

# Use auto_arima to find the best ARIMA model
model = auto_arima(train['Electricity_Consumption'], start_p=1, start_q=1,
                   max_p=3, max_q=3, m=12,
                   start_P=0, seasonal=True,
                   d=1, D=1, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

# Summarize the model
print(model.summary())
#%%
# Make predictions on the test set
predictions = model.predict(n_periods=len(test),index=test.index)
test['Predictions'] = predictions.values

# Calculate and print the MSE
mse = mean_squared_error(np.array(test['Electricity_Consumption']), predictions.values)
print(f'The Mean Squared Error of the forecasts on the test set is: {mse}')

from sklearn .metrics import r2_score

print(r2_score(test['Electricity_Consumption'], predictions.values))

#%%
# Forecast up to 2030
# Assuming the DataFrame df has a DateTime index

model_refit = model.fit(df['Electricity_Consumption'])
# Define the start and end dates
# Define the start date for the forecast as the day after the last date in the historical data
start_date = df.index[-1] + pd.Timedelta(days=1)

# Define the end date for the forecast period
end_date = '2060-01-01'

# Generate the date range for the forecast
forecast_index = pd.date_range(start=start_date, end=end_date, freq='MS')

# Make predictions over the forecast period and get confidence intervals
forecast_values = model_refit.predict(n_periods=len(forecast_index))

# Create a Series for the forecasted values
forecast_series = pd.Series(forecast_values.values, index=forecast_index)
#%%
# Plot the historical data, forecast, and confidence intervals
plt.figure(figsize=(14, 7))
plt.plot(df['Electricity_Consumption'], label='Historical',linewidth=2)
plt.plot(forecast_series, label='Forecast', color='red',linewidth=2)

plt.title('Electricity Consumption Forecast to 2060', fontsize=18, fontweight='bold', fontname='Arial')
plt.xlabel('Date', fontsize=14, fontweight='bold', fontname='Arial')
plt.ylabel('Electricity Consumption (Twh)', fontsize=14, fontweight='bold', fontname='Arial')
plt.legend(fontsize=12)
plt.gca().fill_between(forecast_series.index, forecast_series.values, alpha=0.1, color='red')
plt.grid(True, linestyle='--', alpha=0.5)

# 调整刻度标签的字体大小和样式
plt.xticks(fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')

# 调整图形边框线的样式
plt.gca().spines['top'].set_linestyle('--')
plt.gca().spines['right'].set_linestyle('--')

# 自动调整布局
plt.tight_layout()
plt.show()

# Print the forecast for December 2030
print(f"The forecasted electricity consumption in 2060-01-01 is {forecast_series[-1]:.2f}")

#