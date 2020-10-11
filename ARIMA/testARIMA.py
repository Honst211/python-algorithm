import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Subsetting the dataset
# Index 11856 marks the end of year 2013
df = pd.read_csv('jetrail.csv', nrows=11856)

print(type(df))

# Creating train and test set
# Index 10392 marks the end of October 2013
train = df[0:10392]
test = df[10392:]

# Aggregating the dataset at daily level
df['Timestamp'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')  # 4位年用Y，2位年用y
df.index = df['Timestamp']
df = df.resample('D').mean()  # 按天采样，计算均值

train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
train.index = train['Timestamp']
train = train.resample('D').mean() #

test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%d-%m-%Y %H:%M')
test.index = test['Timestamp']
test = test.resample('D').mean()

# Plotting data
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
import seaborn as sns
sns.set(color_codes=True)
sns.set()
# fit1.plot_diagnostics(figsize=(16, 12))
plt.show()

# 残差分析 正态分布 QQ图线性
fit1.plot_diagnostics(figsize=(16, 12))
plt.show()
