# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import datetime
import re
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pmdarima
from pmdarima import auto_arima
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

# Clean the data
def main_time_series(df):
    # group information and drop na values
    df = data_clean(df)
    # place data into desired format 
    df_arima, df_sarima = data_wrangling(df)

    return df_arima, df_sarima

# clean data from na values 
def data_clean(df):
    df.dropna(subset=['OrderDate'], inplace=True)
    
    return df

# change date into desired format and group
def data_wrangling(df):
    # change creation date to be x days ago from today (dec 27)
    df = creation_date_modifier(df)
    # makes sure that different products in same order are considered 1 order
    df = df.groupby('OrderNum').agg({"OrderDate":"first", "Total Cost":"sum"})
    # group by date, sum # products bought and total cost of them (revenue) 
    df = df.groupby('OrderDate').agg({"OrderDate":"count", "Total Cost":"sum"})
    # set index to OrderDate, rename to Date
    df.index.names = ['Date']
    # rename OrderDate column (# of products ordered on given day) to Amount
    df = df.rename(columns={df.columns[0]:'Amount'})
    # insert 0 Total Cost for days with no sales 
    df = zero_sales(df)
    # organize sales into weekly info (week 1, week 2, ..., week n)
    df = weeky_format_helper(df)
    # organize sales into monthly info (month 1, 2, ..., n)
    df1 = monthly_format_helper(df)
    # decompose data to check for trends
    decompose(df1)
    # organize sales into quarterly info
    df2 = quarterly_format_helper(df)
    # split data into training and testing sets
    df, df_train, df_test = data_split(df)
    # make data stationary 
    #df, df_train = stationary(df, df_train)
    # check for ARIMA parameters 
    df = lag_corr(df, df_train)
    # stationarity test 
    df = adfuller_test(df, df_train)
    # run regression
    df_arima = arima(df, df_train, df_test)
    # run regression with SARIMA
    df_sarima = sarima(df, df_train, df_test)
    return df_arima, df_sarima


# insert zero rows for days with 0 sales - create new df and merge it 
def zero_sales(df):
    # period of first to last date on df, symbolic numbers
    start_date = '2005-01-17'
    end_date = '2009-12-03'
    # create time index 
    date_range = pd.date_range(start=start_date, end=end_date, freq='D').date
    # change df index format to datetime 
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce').date
    # create df with this index 
    dates_df = pd.DataFrame(index=date_range)
    # merge dates df with new one 
    df = dates_df.merge(df, left_index=True, right_index=True, how='left')
    # fill with 0s
    df['Amount'].fillna(0, inplace=True)
    # change index name back to Date
    df.index.names = ['Date']
    # plot daily info
    main_plot(df)
    return df

# decompose and plot data in order to derive trends 
def decompose(df):
    
   # Convert the index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # fill missing values 
    df['Total Cost'].fillna(method='ffill', inplace=True)
        
    df = df.dropna(subset=['Total Cost'])
    decompose = sm.tsa.seasonal_decompose(df['Total Cost'], model='additive',extrapolate_trend='freq')
    fig = decompose.plot()
    fig.set_size_inches(14,7)
    plt.show()
        
# organize data into weekly sales by summing daily sales for 7 day cycles 
def weeky_format_helper(df):
    # change index back to datetime index 
    df.index = pd.to_datetime(df.index)
    # resample to weekly basis (week starts monday)
    df = df.resample('W-Mon').sum()
    # plot weekly 
    main_plot(df)
    # return index back to datetime.date
    df.index = df.index.date
    return df

# organize data into weekly sales by summing daily sales for 30/31 day cycles 
def monthly_format_helper(df):
    # change index back to datetime index 
    df.index = pd.to_datetime(df.index)
    # resample to weekly basis (week starts monday)
    df = df.resample('M').sum()
    # plot weekly 
    main_plot(df)
    # return index back to datetime.date
    df.index = df.index.date
    return df

# organize data into quarterly sales by summing quarterly sales
def quarterly_format_helper(df):
    # change index back to datetime index 
    df.index = pd.to_datetime(df.index)
    # resample to quarterly basis
    df = df.resample('Q').sum()
    # plot quarterly
    main_plot(df)
    # return index back to datetime.date
    df.index = df.index.date
    return df

# split data into testing and training set 
def data_split(df):
    # 2/3 train , 1/3 test
    df_train = df[:400]
    df_test = df[200:]
    return df, df_train, df_test

# run adfuller test to check for stationarity 
def adfuller_test(df, df_train):
    # Ho = non-stationary - null hyp
    # H1 = stationary - reject null hyp 
    sales = df_train['Total Cost']
    # drop null values 
    na_indices = sales.index[sales.isnull()].tolist()
    sales = sales.dropna()
    # run test on sales revenue column
    result = adfuller(sales)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):df['Total Cost'].fillna(method='ffill', inplace=True)
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print('reject the null hypothesis - stationary data') # therefore it is stationary 
    else:
        print('accept the null hypothesis - non stationary')
    return df

# run acf and pacf functions: test for stationarity
def lag_corr(df, df_train):
    # nlags parameter to reduce number of lags collected
    #nlags = min(15, int(0.5 * len(df_train['Total Cost'])))
    sales = df_train['Total Cost']
    sales = sales.dropna()
    
    acf_original = plot_acf(sales, lags=10)
    
    pacf_original = plot_pacf(sales, lags=10)
    
    return df

# use differentiating to make it stationary
def stationary(df, df_train):
    df_diff = df_train.diff()
    # find null values and drop them from both df_diff and df 
    null_values = df_diff.index[df_diff['Total Cost'].isnull()].tolist()
    df_diff.drop(null_values)
    df.drop(null_values)
        
    df_diff.plot()
    
    adfuller_test(df, df_diff)
    
    return df, df_diff 

# run arima model
def arima(df, df_train, df_test):
    # run ARIMA(p,d,0) = ARIMA(18,d,0)
    model = ARIMA(df_train['Total Cost'], order=(0,0,5))
    model_fit = model.fit()
    print(model_fit.summary())
    
    # plot model attributes (residuals plot)
    
    residuals = model_fit.resid[1:]
        
    fig, ax = plt.subplots(1,2, figsize=(10,15))
    
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(title='Density', kind='kde', ax=ax[1])
    
    plt.show()
    
    # predictions 
    forecast = model_fit.forecast(steps=len(df_test))
    
    df['forecast'] = [None]*(len(df_train)) + list(forecast)
    df.drop(columns=['Amount'], inplace=True)
    main_plot_forecast(df)
        
    # test (sum of prediction revenue vs sum of actual revenue)
    
    forecast_rev = sum(list(forecast))
    
    actual_rev = sum(df_test['Total Cost'])
    
    print(forecast_rev)
    print(actual_rev)
    
    return df

# run seasonal arima (sarima) model
def sarima(df, df_train, df_test):
    
    # choose order for sarima model
    my_order = (1,1,1)
    my_seasonal_order = (0,1,0,52)
    # run SARIMA
    model = sm.tsa.statespace.SARIMAX(df_train['Total Cost'], order=my_order, seasonal_order=my_seasonal_order)
    # fit model
    result = model.fit()
    # make predictions
    start = df.shape[0] - df_test.shape[0] + 1
    end = df.shape[0]
    forecast = result.predict(start=start, end=end)
    # add forecast to df
    df['forecast'] = [None]*(len(df_train)) + list(forecast)
    df.drop(columns=['Amount'], inplace=True)
    main_plot_forecast(df)
    
    return df
    

def main_plot(df):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    axes[0].plot(df.index, df['Amount'])
    axes[1].plot(df.index, df['Total Cost'])
    plt.show()

def main_plot_forecast(df):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    ax.plot(df.index, df['Total Cost'])
    ax.plot(df.index, df['forecast'])


if __name__ == '__main__':
    print('please run from main file')

