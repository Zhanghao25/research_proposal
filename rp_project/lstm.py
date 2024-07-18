# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:58:45 2024

@author: rafa
"""
#%%
import pandas as pd
from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
# Load the dataset, skipping the first 27 rows and taking only the first column of actual data
df = pd.read_excel('发电量.xlsx', skiprows=27, usecols=[0,2])
df.columns = ['Date', 'Coal-fired power']
df['Date'] = pd.to_datetime(df['Date'],format="%Y-%m-%d")
df.index=df.Date#将其索引变为时间
df.drop(columns='Date',axis=1,inplace=True)
plt.figure()
plt.plot(df)
plt.show()
#%%
# 绘制时间序列图
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Coal-fired power'])
plt.title('Electricity Generation over Time')
plt.xlabel('Date')
plt.ylabel('Coal-fired Power Generation')
plt.grid(True)
plt.show()

# 绘制季节性图
df['Month'] = df['Date'].dt.month
monthly_average = df.groupby('Month')['Coal-fired power'].mean()
plt.figure(figsize=(10, 6))
plt.bar(monthly_average.index, monthly_average.values)
plt.title('Monthly Average Electricity Generation')
plt.xlabel('Month')
plt.ylabel('Average Coal-fired Power Generation')
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(axis='y')
plt.show()

# 绘制自相关图和偏自相关图
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.figure(figsize=(12, 6))
plot_acf(df['Coal-fired power'], lags=50)
plt.title('Autocorrelation Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(df['Coal-fired power'], lags=50)
plt.title('Partial Autocorrelation Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.grid(True)
plt.show()

# 绘制箱线图
plt.figure(figsize=(8, 6))
plt.boxplot(df['Coal-fired power'])
plt.title('Boxplot of Coal-fired Power Generation')
plt.ylabel('Coal-fired Power Generation')
plt.grid(True)
plt.show()


#%%
# First, we need to install the necessary packages for building an LSTM model.
# We will use TensorFlow and Keras for this purpose.
# The installation command would typically look like this:
# !pip install tensorflow

# However, we cannot install packages in this environment, so we'll proceed assuming the packages are installed.

# Importing the necessary libraries for building an LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

# Data preprocessing

# Normalize the feature data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Coal-fired power']].values)

# Define the LSTM model parameters
n_input = 12  # Length of the output sequences (in number of timesteps)
n_features = 1  # Number of features we want to predict (since we are predicting 'Coal-fired power', it's 1)

# Split the data into training and test sets
train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

# Generate time series sequences
generator = TimeseriesGenerator(train_data, train_data, length=n_input, batch_size=1)

#%%

# 定义更复杂的LSTM模型
model = Sequential()
# 第一个LSTM层，返回序列以供下一个LSTM层使用
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
model.add(Dropout(0.2))  # Dropout层，丢弃20%的神经元

# 第二个LSTM层，不需要返回序列
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))  # 又一个Dropout层

# 输出层，预测单个值
model.add(Dense(1))

# 模型摘要
print(model.summary())

# 编译模型，使用adam优化器和均方误差损失函数
model.compile(optimizer='adam', loss='mse')

# 训练模型，使用之前定义的时间序列生成器
model.fit(generator, epochs=20)  # 增加迭代次数以适应更复杂的模型

#%%


generator1 = TimeseriesGenerator(test_data, test_data, length=n_input, batch_size=1)
# Use the model to make predictions on the test data
predicted_data = model.predict_generator(generator1)

# 反归一化预测结果
predicted_data_original_scale = scaler.inverse_transform(predicted_data)

# 获取对应的实际值（考虑到 TimeseriesGenerator 的长度）
actual_data = test_data[:-n_input]

# 反归一化实际值
actual_data_original_scale = scaler.inverse_transform(actual_data)

# 可视化比较
test_index = df.index[-len(test_data):][:-n_input]
plt.figure(figsize=(12, 6))
plt.plot(test_index,actual_data_original_scale, label='Actual Data')
plt.plot(test_index,predicted_data_original_scale, label='Predicted Data', linestyle='--')
plt.title('Comparison of Actual and Predicted Data')
plt.xlabel('Year')
plt.ylabel('Coal-fired power')
plt.legend()
plt.show()



mse = mean_squared_error(actual_data_original_scale, predicted_data_original_scale)
print('Mean Squared Error (MSE):', mse)


#%%
# 从 scaled_data 中提取所有数据作为初始序列
# 假设scaled_data是你的归一化数据，n_features是特征数量
# 假设你已经有一个训练好的模型和一个scaler用于逆变换

# 使用最后几个时间点的数据作为初始序列

current_batch = TimeseriesGenerator(scaled_data[-n_input:], scaled_data[-n_input:], length=n_input, batch_size=1)

# 确定需要预测到2030年的时间步数
future_steps = 12 * (2030 - df.index[-1].year) + (12 - df.index[-1].month + 1)

# 预测未来的所有时间步
future_predictions = []

for i in range(future_steps):
    # 获取下一个时间步的预测值
    current_pred = model.predict(current_batch)[0]
    
    # 将预测值添加到future_predictions数组中
    future_predictions.append(current_pred)
    
    # 更新current_batch以包含最新预测值
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# 逆变换预测结果为原始数据规模
future_predictions_original = scaler.inverse_transform(np.array(future_predictions).reshape(-1, n_features))

# 生成未来日期的索引
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=future_steps, freq='M')

# 绘制未来预测值的折线图
plt.figure(figsize=(15, 5))
plt.plot(future_dates, future_predictions_original, label='Future Predicted')
plt.title('Future Predictions up to 2030')
plt.xlabel('Date')
plt.ylabel('Value')  # 修改为你的数据对应的值的标签
plt.legend()
plt.show()

#%%
