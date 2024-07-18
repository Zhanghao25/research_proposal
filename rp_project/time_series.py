# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:42:25 2024

@author: rafa
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Since we cannot install packages, we'll use statsmodels instead of pmdarima which might not be available.
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose


#%%
# 假设您的数据已经被加载到一个名为 df 的 pandas DataFrame 中
# df['Year'] 是年份，df['Value'] 是相应年份的值
import pandas as pd
from prophet import Prophet
df = pd.read_excel('能源生产发电量 1987-2023 avg.xlsx',skiprows=27)
df.columns = ['Date', 'Electricity_Production']
df['Date'] = pd.DatetimeIndex(df['Date'])

df = df.rename(columns={'Date': 'ds',
                        'Electricity_Production': 'y'})
df.head()

m = Prophet()
m.fit(df)
ax = df.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Monthly Number of Airline Passengers')
ax.set_xlabel('Date')

plt.show()
#%%
my_model = Prophet(interval_width=0.95)
my_model.fit(df)
future_years = 6  # 要添加的年数
future_dates = m.make_future_dataframe(periods=future_years, freq='Y')
future_dates['ds'] = future_dates['ds'].dt.strftime('%Y-%m-01')
#%%
forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
my_model.plot(forecast, uncertainty=True)
my_model.plot_components(forecast)
#%%




# Load the data
# Note: The actual data loading step is not shown here as it cannot be executed in this environment.
# The code below assumes that the dataframe 'df' has already been created and preprocessed as per the instructions.

# Perform seasonal decomposition to guess initial values for p, d, q
decomposition = seasonal_decompose(df['Electricity_Production'], model='additive', period=1)
decomposition.plot()
plt.show()

# Guess starting values for p, d, q by looking at the ACF and PACF plots
# Here we are assuming non-seasonal data with a period of 1 year
# If the data was seasonal, we would look for seasonality and set the 'm' parameter in auto_arima

# Fit the SARIMAX model using statsmodels (as a substitute for pmdarima's auto_arima)
# Note: we will start with a basic SARIMAX model without seasonal components
# In a local environment, you should use auto_arima to find the best parameters

model = SARIMAX(df['Electricity_Production'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)

# Summary of the model
print(model_fit.summary())

# Forecast the next 7 years to get to 2030
forecast = model_fit.get_forecast(steps=7)
forecast_index = pd.date_range(df.index[-1], periods=8, freq='A-DEC')[1:]  # Exclude the last known data point
forecast_df = pd.DataFrame({'Forecast': forecast.predicted_mean}, index=forecast_index)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df['Electricity_Production'], label='Actual')
plt.plot(forecast_df['Forecast'], label='Forecast', color='red')
plt.title('Electricity Production Forecast to 2030')
plt.legend()
plt.show()

# Print the forecast for 2030
forecast_2030 = forecast_df.iloc[-1]
print(f"The forecasted electricity production in 2030 is {forecast_2030['Forecast']:.2f} billion kWh")

#%%
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('能源生产发电量 1987-2023 avg.xlsx', skiprows=27)
df.columns = ['Date', 'Electricity_Production']
df['Date'] = pd.DatetimeIndex(df['Date'])

df = df.rename(columns={'Date': 'ds',
                        'Electricity_Production': 'y'})

# 创建并拟合Prophet模型
my_model = Prophet(interval_width=0.95)
my_model.fit(df)

# 创建要预测的未来日期，以年为单位
future_years = 36
future_dates = pd.date_range(start=df['ds'].max(), periods=future_years+1, freq='Y')
future_dates = pd.DataFrame({'ds': future_dates.strftime('%Y-%m-01')})

# 预测未来数据
forecast = my_model.predict(future_dates)

# 绘制预测结果
plt.figure(figsize=(12, 8))

# 绘制原始数据
plt.plot(df['ds'], df['y'], label='Actual', color='blue', marker='o', markersize=5)

# 标明置信区间
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.5, label='Confidence Interval')

# 绘制新预测的年份
new_predictions = forecast[forecast['ds'].dt.year > df['ds'].dt.year.max()]
plt.plot(new_predictions['ds'], new_predictions['yhat'], label='Predicted (New Years)', color='red', marker='o', markersize=5)

plt.xlabel('Date')
plt.ylabel('Electricity Production (TWh)')
plt.legend()
plt.title('Electricity Production Forecast')
plt.show()
