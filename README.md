# Time Series Analysis

## Description

Conducted a time series analysis using hypothetical sales data. The goal was to predict future sales based on past experiences. 

## Data Preparation

I started by preparing the data, which included extracting and transforming the necessary columns: Order Date, Products Ordered and Price.
By applying various transformations, I turned this into a revenue by day dataframe. Examples of this include aggregating daily revenue, or 
filling zero-sale days in the dataframe.

The second component of data preparation is data exploration. I transformed the data into datetime objects
and divided it into quarters, months and weeks. I should note that I kept weeks, as it ensured I didn't miss essential patterns. 

Finally, I split the data into training and testing sets, chronologically. 

## The models 

### Time Series Decomposition

To get a closer understanding of the data, I began with a decomposition. I divided my data into trend, seasonality and residuals.
The trend, which indicates the long term direction of the data, indicated a progressive increase in sales. 
At the same time, the seasonal component indicates that there is a strongly marked seasonal component in the data. Finally, the 
residual category shows unexplained behavior. In this case, it wasn't high.

Math note: the trend is estimated by calculating a simple moving average of the data. This 'smoothes' the data by dying down
long term variations. Next, the seasonal component is obtained by removing the subtracting the trend from the original data -- in the
additivie model. Finally, the residuals are found by removing both trend and seasonality from the original data. 

![ts_fig1](https://github.com/user-attachments/assets/b82f85be-e93c-4094-8c1e-a0ec34d1a274)


### ARIMA

The ARIMA model is the combination of an Autoregression term (AR), an Integration term (I) and a Moving Average Term (MA). Essentially, 
the AR term indicates the relation between a point and lagged points in the dataset (previous values). The MA term, as explained above,
explains the deviation of a point from a moving average model applied to previous points. Finally, the integration term refers to removing
seasonality and residual erros (mentioned in the decomposition) from the model. 

Before applying this model, you must find the ideal coefficients for the AR and MA lag terms. You can use the Autocorrelation Function (ACF)
to plot the correlation between the current and past values. In this case, the point at which the the correlation is the expected optimal 
MA lag term: 5. The ACF also indicates if a dataset is stationary; however, since we conduct an adfuller test, we already know this to be the case.
Moving on, the Partial Autocorrelation Function (PACF) measures the correlation between the current and past individual values. This means that
the value we select will tell us the period in which a point is influenced by past points: in  this case, two. 

![Screenshot 2024-08-29 180018](https://github.com/user-attachments/assets/6141dc04-bafe-4222-a968-1d618bb19349)


Plotting the ARIMA model result, it was clearly not a great fit. I realized that, since the x axis only goes to 10, the PACF fails to capture the seasonal effect. 
Our Autoregression term is incorrect. If we we're to increase the PACF x-axis further, we would see a strong correlation between a given point and a past term: the seasonal component. 
This wouldn't be a clean solution, however. This brings me to our third (and most succesful) model: SARIMA. 

![Screenshot 2024-08-29 180121](https://github.com/user-attachments/assets/095502d3-9e58-4a1c-9696-605f3b7f1871)


### SARIMA

The SARIMA model includes everything the ARIMA does with the added benefit of a seasonal component. In this case, estimating based on the decomposition model, I added
a seaonal autoregressive term of 52 (weeks). In this example, this means that the point in our series will be reduced by the term that came 52 weeks ahead. 

![sarima_fig2](https://github.com/user-attachments/assets/dbd72fc5-90b3-4f0a-b37a-c1204c96054e)

