import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm

# read data, convert the datetime column ans set it as the index
df = pd.read_csv("EMHIRESPV_TSh_CF_Country_19862015.csv", parse_dates = ["Date"], index_col = ["Date"])

df.columns

name_map = {
    'AL': "Albania", 'AT': "Austria", 'BA': "Bosnia", 'BE': "Belgium", 'BG': "Bulgaria", 'CH': "Switzerland", "CY": "Cyprus", 
    'CZ': "Czechia", 'DE': "Germany", 'DK': "Denmark", 'EE': "Estonia", 'ES': "Spain", 'FI': "Finland", 'FR': "France", 
    'EL': "Greece", 'HR': "Croatia", 'HU': "Hungary", 'IE': "Ireland", 'IT': "Italy", 'LT': "Lithuania", 'LU': "Luxembourg", 
    'LV': "Latvia", 'ME': "Montenegro", 'MK': "North Macedonia", 'NL': "Netherlands", 'NO': "Norway", 'PL': "Poland", 
    'PT': "Portugal", 'RO': "Romania", 'RS': "Serbia", 'SI': "Slovenia", 'SK': "Slovakia", 'SE': "Sweden", 'XK': "Kosovo", 
    'UK': "United Kingdom"
}

# exploratory plot for the comparision of solar power generation for four countries in Europe
fig, ax = plt.subplots(figsize = (20, 10))
df.loc["2000":"2015"][["BE","FR","DE","UK"]].resample("y").sum().plot(marker = ".", linestyle = "--", ax = ax)
ax.set_xlabel('Time', fontsize=20)
ax.set_ylabel("Power generation (Gwh)", fontsize = 20)
ax.tick_params(axis='both', labelsize=15)
ax.legend((name_map["BE"], name_map["FR"], name_map["DE"], name_map["UK"]), fontsize = 15, loc = "upper right")
plt.show()

# Resample daily data
df_daily = df[["BE","FR","DE","UK"]].resample("D").sum()

# subplot the daily power generation
df_daily.plot(subplots = True, figsize = (20, 10))
[ax.legend(loc=1) for ax in plt.gcf().axes]
plt.show()

# Resample weekly data
df_weekly = df[["BE","FR","DE","UK"]].resample("W").sum()

# subplot the weekly power generation
df_weekly.plot(subplots = True, figsize = (20, 10))
[ax.legend(loc=1) for ax in plt.gcf().axes]
plt.show()

# Resample monthly data
df_monthly = df[["BE","FR","DE","UK"]].resample("M").sum()

# subplot the monthly power generation
df_monthly.plot(subplots = True, figsize = (20, 10))
[ax.legend(loc=1) for ax in plt.gcf().axes]
plt.show()

# Resample yearly data
df_yearly = df[["BE","FR","DE","UK"]].resample("Y").sum()

# subplot the yearly power generation
df_yearly.plot(subplots = True, figsize = (20, 10))
[ax.legend(loc=1) for ax in plt.gcf().axes]
plt.show()

# Take monthly data from France as an example, first run the augmented dicky-fuller test function to check the stationality of time series
result = adfuller(df_monthly["FR"])

# based on the p-value, we reject the null hypothesis that the time series is non-stationary. Therefore the time series is stationary. 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Will use ARIMA model, so need to create ACF and PACF to determine the best model orders
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
 
# Plot the ACF of df
plot_acf(df_monthly["FR"], lags=30, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(df_monthly["FR"], lags=30, zero=False, ax=ax2)
plt.show()

# Use Akaike information criterion (AIC) and Bayesian information criterion (BIC), to test different combinations of model orders
import warnings
warnings.filterwarnings('ignore')

order_aic_bic=[]

# Loop over p values from 0-15
for p in range(15):
  # Loop over q values from 0-15
    for q in range(15):
        try:
            model = SARIMAX(df_monthly["FR"], order=(p,0,q))
            results = model.fit()
            order_aic_bic.append((p,q,results.aic, results.bic))
        except:
            order_aic_bic.append((p,q,np.nan, np.nan))

# Construct DataFrame from order_aic_bic, show the order of increasing AIC and increasing BIC
order_df = pd.DataFrame(order_aic_bic, 
                        columns=["p", "q", "AIC", "BIC"])

print(order_df.sort_values("AIC"))
print(order_df.sort_values("BIC"))

# Based on the results, we can try choosing p = 7, and q = 10, so create the correpsonding model and fit with data
model1 = SARIMAX(df_monthly["FR"], order=(7,0,10))
results1 = model1.fit()
print(results1.summary())

# Create the diagostics plots, and it can been there might exist seasonal patterns based on the standardized residuals.
results1.plot_diagnostics(figsize = (20,10))

# So next perform and plot additive decomposition of time series for exploring trends, seasonal patterns, and residuals. 
decomp = seasonal_decompose(df_monthly["FR"], freq = 12, model='additive')

fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(20,10))
decomp.trend.plot(ax=ax1)
decomp.resid.plot(ax=ax2)
decomp.seasonal.plot(ax=ax3)
plt.show()

# Also perform and plot multiplicative decomposition of time series for exploring trends, seaonsal patterns, and residuals. 
decomp = seasonal_decompose(df_monthly["FR"], freq = 12, model='multiplicative')

fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(20,10))
decomp.trend.plot(ax=ax1)
decomp.resid.plot(ax=ax2)
decomp.seasonal.plot(ax=ax3)
plt.show()

# The pmdarima package is a powerful tool to help choose the model orders with seasonality being considerd.
model2 = pm.auto_arima(df_monthly["FR"], seasonal=True, m=12, d = 1, D=1, max_p=3, max_q=3, trace=True, error_action='ignore', trend = "c", suppress_warnings=True) 

# Based on the results, the ARIMA(3,1,0)(2,1,1)[12] is selected as the relatively best one
print(model2.summary())

# fit the model with the determined orders
model3 = SARIMAX(df_monthly["FR"], order=(3,1,0), seasonal_order=(2,1,1,12), trend="c")
results3 = model3.fit()

# generating one-step-ahead predictions, extract prediction mean, get confidence intervals, for the latest 36 months
one_step_forecast = results3.get_prediction(start=-36)

mean_forecast = one_step_forecast.predicted_mean

confidence_intervals = one_step_forecast.conf_int()

lower_limits = confidence_intervals.loc[:,'lower FR']
upper_limits = confidence_intervals.loc[:,'upper FR']

# plot the one-step-ahead prediction results
plt.figure(figsize=(20,10))
plt.plot(df_monthly.index, df_monthly.FR, label='observed')
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')
plt.xlabel('Time', fontsize=20)
plt.ylabel("Power generation (Gwh)", fontsize = 20)
plt.title("France Solar PV power generation (1986-2015)", fontsize = 22)
plt.tick_params(axis='both', labelsize=15)



