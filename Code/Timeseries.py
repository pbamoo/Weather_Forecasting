#!/usr/bin/env python
# coding: utf-8

# ## Goal : Create temperature forecast.
# 
# - Understand the purpose of Time Series Analysis
# - Understand the concept of time dependence in time series data
# - Understand how to use AR and ARIMA models in order to predict the future
# 
# Data and metadata available at http://www.ecad.eu
# 
# * FILE FORMAT (MISSING VALUE CODE IS -9999):
# * 01-06 SOUID: Source identifier
# * 08-15 DATE : Date YYYYMMDD
# * 17-21 TG   : mean temperature in 0.1 &#176;C
# * 23-27 Q_TG : Quality code for TG (0='valid'; 1='suspect'; 9='missing')
# 
# This is the blended series of station GERMANY, BREMEN (STAID: 42)

# ### Import necessary libraries

# In[1]:

from subprocess import call
import json
import plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.express as px
init_notebook_mode(connected=True)
import plotly
plotly.offline.init_notebook_mode()

# hide warnings
import warnings
warnings.simplefilter("ignore")

plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"
plt.figure(figsize=[15,6])


### Load & Explore Data

# read the csv file into a DataFrame 
data = pd.read_csv('../data/TG_STAID000042.txt', sep=',', comment='%', skiprows=18, skipfooter=10)

##format and rename columns
data.columns = data.columns.str.strip() 
cols = ['source_id', 'date', 'temp', 'q_temp']
data.columns = cols

#Convert date to DateTime format and set as index
data['date'] = pd.to_datetime(data['date'].astype(str), format='%Y%m%d')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data.set_index('date', inplace=True)

#convert mean temperature in 0.1 °C
data['temp'] = data['temp'] * 0.1

#drop missing / suspect values
data.drop(data.loc[data['q_temp']>=1].index, inplace=True)

# Consider data from 1950 due to lots of missing data from 1876-1946
df = data[ data.year >= 1950 ]

#Visualize Temperature Time Series
fig = px.line(x=df.index, y=df.temp)
fig.update_layout(title_text="Temperature Time Series", xaxis_title="Date", yaxis_title="Temperature, C")
fig.update_xaxes(rangeslider_visible=True, 
rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")])))
plotly.offline.plot(fig, filename=r'../Images/1_Time_Series.png')
#fig.write_image("../Images/1_Time_Series.png")

# use pivot table to observe & compare the mean temperature for months for various years
dt = pd.pivot_table(df, index='month', values='temp', columns='year')
sns.heatmap(dt, cmap='coolwarm', robust=True)
plt.title("Average Tempearture in Bremen from 1950 to 2020")


# Check for Stationarity

# 1. Check components
from statsmodels.tsa.seasonal import seasonal_decompose
sdr = seasonal_decompose(df['temp'], model='additive', extrapolate_trend='freq', freq=365)
fig = sdr.plot()
fig.savefig('../Images/2_componenents.png')

# 2. Augmented Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller

print("Performing Augmented Dickey-Fuller Test to confirm stationarity...")
result = adfuller(df.temp.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
p = result[1]
if (p > 0.05):
    print("Time Series is NOT Stationary, since p-value > 0.05")
    df = df.diff()  # differencing to make data stationary
else:
    print("Time Series is Stationary, since p-value <= 0.05")

# Autocorrelation and partial autocorrelation functions
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
plot_acf(df.temp.tolist(), lags=30, ax=axes[0])
plot_pacf(df.temp.tolist(), lags=30, ax=axes[1])
fig.savefig('../Images/3_pacf.png')


# Create train data
train_df = df["temp"][:-13]
date = df.index[:-13]


# 1.  AR model

# with statsmodel 
from statsmodels.tsa.ar_model import AR
ar = AR(train_df, dates=date).fit(maxlag=52, ic='aic')

# prediction is 
ar_predict = ar.predict('2019-10-22','2020-10-21')

# Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(name="Raw Data", x=df.index, y=df.temp))
fig.add_trace(go.Scatter(name="AR model Prediction", x=ar_predict.index, y=ar_predict))
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(title_text="AR MODEL", xaxis_title="Date", yaxis_title="Temperature, C")
plotly.offline.plot(fig, filename=r'../Images/4_AR.png')


# 2. ARMA Model
# with statsmodel, aic check of params
from statsmodels.tsa import stattools as st
from statsmodels.tsa.arima_model import ARMA, ARIMA, ARMAResults

st.arma_order_select_ic(train_df, ic='aic')
arma = ARMA(train_df, order=[3,2]).fit(maxlag=4, ic='aic', dates=date)
arma_predict = arma.predict('2019-10-22','2020-10-21')

# Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(name="Raw Data", x=df.index, y=df.temp))
fig.add_trace(go.Scatter(name="ARMA model Prediction", x=arma_predict.index, y=arma_predict))
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(title_text="ARMA MODEL", xaxis_title="Date", yaxis_title="Temperature, C")
plotly.offline.plot(fig, filename=r'../Images/5_ARMA.png')


# 3. ARIMA MODEL
# predict with statsmodel, p,q are same as ARMA.
arima = ARIMA(train_df, order=[3,0,2],).fit(ic='aic', dates=date)
arima_predict = arima.predict('2019-10-22','2020-10-21')

# Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(name="Raw Data", x=df.index, y=df.temp))
fig.add_trace(go.Scatter(name="ARIMA model Prediction", x=arima_predict.index, y=arima_predict))
fig.update_layout(title_text="ARIMA MODEL", xaxis_title="Date", yaxis_title="Temperature, C")
fig.update_xaxes(rangeslider_visible=True)
plotly.offline.plot(fig, filename=r'../Images/6_ARIMA.png')

# Visualization
fig = go.Figure()
fig.add_trace(go.Scatter(name="Raw Data", x=df.index, y=df.temp))
fig.add_trace(go.Scatter(name="ARIMA model Prediction", x=arima_predict.index, y=arima_predict))
fig.update_layout(xaxis_range=['2017-01-01','2020-12-31'], title_text="ARIMA MODEL")
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(title_text="ARIMA MODEL", xaxis_title="Date", yaxis_title="Temperature, C")
plotly.offline.plot(fig, filename=r'../Images/7_ARIMA.png')

# comparing first 100 predictions with actual values

arima_pred = arima.predict(start=0, end=len(train_df)-1)
plt.plot(list(train_df)[:100], label="Actual")
plt.plot(list(arima_pred)[:100], 'r', label='Predicted')
plt.xlabel("Time (in Months)")
plt.ylabel("Temperature (C)")
plt.title("Actual and Predicted Temperature Values")
plt.legend(loc='upper center', bbox_to_anchor=(1.00, 0.8))