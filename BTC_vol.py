#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


# Define the tickers for MSTR and BTC-USD
tickers = ['BTC-USD', '^SPX', 'GC=F'] # , '^SPX', 'GC=F', '^TYX'

today = datetime.today().strftime('%Y-%m-%d')


# Fetch historical data for both assets
data = yf.download(tickers, start="2010-07-07", end=today)['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Perform correlation analysis
correlation_matrix = returns.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap of Daily Returns")
plt.show()

today


# ## Gold and BTC Correlation

# In[ ]:


rolling_window = 200  # Rolling window size (e.g., n days)
corr_df2 = pd.DataFrame()
corr_df2['correlation'] = returns['GC=F'].rolling(rolling_window).corr(returns['BTC-USD'])
corr_df2.dropna(inplace=True)

corr_df_mean2 = pd.DataFrame()
corr_df_mean2['mean'] = corr_df2.mean()
corr_df_mean2.dropna(inplace=True)

corr_df_ma2 = pd.DataFrame()
corr_df_ma2['ma'] = corr_df2.rolling(window=365).mean()
corr_df_ma2.dropna(inplace=True)

plt.figure(figsize=(9,5))
plt.title('Correlation of BTC & Gold Returns')
plt.plot(corr_df2)
plt.grid(alpha = 0.2)
plt.plot(corr_df_ma2['ma'])
plt.axhline(y = 0.070099, color = 'purple', linestyle = '--', linewidth = 2, label = 'Mean Correlation')
plt.show()


from statsmodels.tsa.stattools import adfuller

# Perform the ADF test on BTC volatility
result = adfuller(corr_df2['correlation'].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Reject the null hypothesis: The correlation is likely mean-reverting.")
else:
    print("Fail to reject the null hypothesis: The correlation is likely non-stationary.")


# ### SPY & BTC Correlation 

# In[ ]:


rolling_window = 200  # Rolling window size (e.g., n days)
corr_df = pd.DataFrame()
corr_df['correlation'] = returns['^SPX'].rolling(rolling_window).corr(returns['BTC-USD'])
corr_df.dropna(inplace=True)

corr_df_mean = pd.DataFrame()
corr_df_mean['mean'] = corr_df.mean()
corr_df_mean.dropna(inplace=True)

corr_df_ma = pd.DataFrame()
corr_df_ma['ma'] = corr_df.rolling(window=365).mean()
corr_df_ma.dropna(inplace=True)

plt.figure(figsize=(9,5))
plt.title('Correlation of BTC & SPY Returns')
plt.plot(corr_df)
plt.grid(alpha = 0.2)
plt.plot(corr_df_ma['ma'])
plt.axhline(y = 0.170099, color = 'purple', linestyle = '--', linewidth = 2, label = 'Mean Correlation')
plt.show()




# Perform the ADF test on BTC volatility
result = adfuller(corr_df['correlation'].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Reject the null hypothesis: The correlation is likely mean-reverting.")
else:
    print("Fail to reject the null hypothesis: The correlation is likely non-stationary.")
            


# ### Statistical Test Upon 2014 - 2020 and 2020 Onwards Mean

# In[ ]:


corr_1 = corr_df.iloc[:1754]
corr_2 = corr_df.iloc[1754:]

corr_1_mean = corr_1.mean()
corr_2_mean = corr_2.mean()
corr_1_mean, corr_2_mean


# In[ ]:


from scipy.stats import mannwhitneyu

# Perform the Mann-Whitney U Test
u_stat, p_value = mannwhitneyu(corr_1, corr_2)

# Explicitly extract the scalar value from the array (if necessary)
p_value = p_value.item()  # Safely converts a single-element array to a scalar

# Set a significance level (commonly 0.05)
alpha = 0.01

# Check for statistical significance
if p_value < alpha:
    print(f"The result is statistically significant (p-value = {p_value:.4f}).")
else:
    print(f"The result is NOT statistically significant (p-value = {p_value:.4f}).")

# Output U statistic for reference
print(f"Mann-Whitney U statistic: {u_stat}")


# ### 30 Day Rolling Vol of n Assets

# In[ ]:


# Calculate rolling volatility (standard deviation) over a 30-day window
rolling_volatility = returns.rolling(window=30).std()

# Plot the rolling volatility
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(rolling_volatility[ticker], label=f"{ticker} 30-Day Volatility")

# Add labels and legend
plt.title("Rolling 30-Day Volatility of Daily Returns")
plt.xlabel("Date")
plt.ylabel("Volatility (Standard Deviation)")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


vol_ma = rolling_volatility['BTC-USD'].rolling(window = 100).mean()


# In[ ]:


btc_vol = rolling_volatility['BTC-USD']
btc_vol = pd.DataFrame(btc_vol)
btc_vol.dropna().head(5)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure 'btc_vol' and 'data' contain relevant time series
btc_price = data['BTC-USD']
log_btc_price = np.log(btc_price)  # Log-transformed BTC price
btc_volatility = btc_vol['BTC-USD']  # BTC volatility series

# Calculate the moving average for BTC volatility
vol_ma = btc_volatility.rolling(window=365).mean()  # 30-day moving average

# Create a figure with twin axes for dual plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot log-transformed BTC price on the primary y-axis
ax1.plot(log_btc_price, color='green', label='Log BTC Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Log BTC Price', color='green')
ax1.tick_params(axis='y', labelcolor='green')

# Create a twin y-axis for BTC volatility
ax2 = ax1.twinx()
ax2.plot(btc_volatility, color='orange', label='BTC Volatility', linestyle='--')
ax2.plot(vol_ma, color='purple', label='Volatility Moving Average (365-Day)', linestyle='-')  # Add moving average line
ax2.axhline(y=0.0328, color='purple', linestyle='--', label='Mean Volatility')  # Add horizontal line for vol mean
ax2.set_ylabel('BTC Volatility', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add vertical lines for halving dates
halving_dates = [pd.Timestamp('2016-07-09'), pd.Timestamp('2020-05-11'), pd.Timestamp('2024-04-19')]
for date in halving_dates:
    ax1.axvline(x=date, color='red', linestyle='--', alpha=0.7, label='Halving Date' if date == halving_dates[0] else "")

# Add a title and grid
plt.title('Log BTC Price, BTC Volatility, and Moving Average with Mean Vol')
ax1.grid(visible=True, linestyle='--', alpha=0.5)

# Add a combined legend
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.85))

# Show the plot
plt.show()


# # Stat Diff Between Cycle 1 and Cycle 2 Vol Means

# In[ ]:


cycle1_vol = btc_vol['BTC-USD'].iloc[600:2200].dropna().sort_values(ascending=True)
cycle2_vol = btc_vol['BTC-USD'].iloc[2200:3600].dropna().sort_values(ascending=True)


# In[ ]:


u_stat, p_value = mannwhitneyu(cycle1_vol, cycle2_vol, alternative='two-sided')
print(f"Mann-Whitney U Test: U-Statistic = {u_stat}, P-Value = {p_value:.4f}")
    
if p_value < 0.05:
    print("Significant difference between volatilities before and after.")
    
else:
        print("No significant difference between volatilities before and after.")


# # Volatility Mean Reversion Tests

# In[ ]:


from hurst import compute_Hc
# Compute the Hurst exponent
H, _, _ = compute_Hc(btc_vol['BTC-USD'].dropna(), kind='price')
print("Hurst Exponent:", H)

if H < 0.5:
    print("Mean-reverting behavior detected.")
elif H == 0.5:
    print("Random walk behavior detected.")
else:
    print("Trending behavior detected.")


# ### Hurst Exponent on Cycle 2

# In[ ]:


# Compute the Hurst exponent
H, _, _ = compute_Hc(btc_vol['BTC-USD'].iloc[2200:3600].dropna(), kind='price')
print("Hurst Exponent:", H)

if H < 0.5:
    print("Mean-reverting behavior detected.")
elif H == 0.5:
    print("Random walk behavior detected.")
else:
    print("Trending behavior detected.")


# ### Hurst Exponent on Cycle 1

# In[ ]:


# Compute the Hurst exponent
H, _, _ = compute_Hc(btc_vol['BTC-USD'].iloc[600:2200].dropna(), kind='price')
print("Hurst Exponent:", H)

if H < 0.5:
    print("Mean-reverting behavior detected.")
elif H == 0.5:
    print("Random walk behavior detected.")
else:
    print("Trending behavior detected.")


# ### Cycle 1 ADF Test

# In[ ]:


from statsmodels.tsa.stattools import adfuller

# Perform the ADF test on BTC volatility
result = adfuller(btc_vol['BTC-USD'].iloc[600:2200].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Reject the null hypothesis: The volatility is likely mean-reverting.")
else:
    print("Fail to reject the null hypothesis: The volatility is likely non-stationary.")


# In[ ]:


result = adfuller(btc_vol['BTC-USD'].iloc[2200:3600].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] < 0.05:
    print("Reject the null hypothesis: The volatility is likely mean-reverting.")
else:
    print("Fail to reject the null hypothesis: The volatility is likely non-stationary.")


# ## Standard Deviation and Z-Scores of Reverting Volatility

# In[ ]:


cycle1_vol = btc_vol['BTC-USD'].iloc[700:2000].dropna()
cycle2_vol = btc_vol['BTC-USD'].iloc[2200:3550].dropna()


# In[ ]:


mean_cycle1 = cycle1_vol.mean()
std_cycle1 = cycle1_vol.std()

mean_cycle2 = cycle2_vol.mean()
std_cycle2 = cycle2_vol.std()


# In[ ]:


cycle1_vol_z = (cycle1_vol - mean_cycle1) / std_cycle1
cycle2_vol_z = (cycle2_vol - mean_cycle2) / std_cycle2


# In[ ]:


import matplotlib.pyplot as plt

# Plot for Cycle 1
plt.figure(figsize=(8, 4))
plt.plot(cycle1_vol.index, cycle1_vol_z, label="Cycle 1 Z-Scores", color="blue")
plt.axhline(y=3, color='red', linestyle='--', label="+3 Std Dev")
plt.axhline(y=2, color='orange', linestyle='--', label="+2 Std Dev")
plt.axhline(y=-2, color='orange', linestyle='--', label="-2 Std Dev")
plt.axhline(y=-3, color='red', linestyle='--', label="-3 Std Dev")
plt.axhline(y=0, color='black', linestyle='-', label="Mean")
plt.title("Bitcoin Cycle 1 Volatility Z-Scores")
plt.xlabel("Time")
plt.ylabel("Z-Score")
plt.legend()
plt.show()

# Plot for Cycle 2
plt.figure(figsize=(8, 4))
plt.plot(cycle2_vol.index, cycle2_vol_z, label="Cycle 2 Z-Scores", color="green")
plt.axhline(y=3, color='red', linestyle='--', label="+3 Std Dev")
plt.axhline(y=2, color='orange', linestyle='--', label="+2 Std Dev")
plt.axhline(y=-2, color='orange', linestyle='--', label="-2 Std Dev")
plt.axhline(y=-3, color='red', linestyle='--', label="-3 Std Dev")
plt.axhline(y=0, color='black', linestyle='-', label="Mean")
plt.title("Bitcoin Cycle 2 Volatility Z-Scores")
plt.xlabel("Time")
plt.ylabel("Z-Score")
plt.legend()
plt.show()


# # Vol and Returns Correlation

# In[ ]:


# Calculate daily returns
btc_returns = returns['BTC-USD'].dropna()

# Correlation between returns and volatility
correlation = btc_returns.corr(btc_volatility)
print(f"Correlation between BTC Returns and Volatility: {correlation:.4f}")


# # Vol % Change and Returns Correlation

# In[ ]:


btc_returns = btc_returns

btc_vol_pct = btc_volatility.pct_change().dropna()
correlation = btc_returns.corr(btc_vol_pct)
print(f"Correlation between BTC Returns and Volatility % Change: {correlation:.4f}")


# # Cycle High and Low Volatility Changes

# ### Distribution of Vol

# In[ ]:


# Ensure 'btc_vol' contains the Bitcoin volatility series
btc_volatility = btc_vol['BTC-USD']  # Replace 'BTC-USD' with the correct column name if different

# Plot the distribution of btc_vol
plt.figure(figsize=(12, 6))
sns.histplot(btc_volatility, kde=True, bins=50, color='blue', alpha=0.7, label='BTC Volatility')

# Add labels and a title
plt.title('Distribution of BTC Volatility', fontsize=16)
plt.xlabel('Volatility', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()

from scipy.stats import moment

# Calculate moments of the observed data
observed_mean = btc_volatility.mean()
observed_variance = btc_volatility.var()
observed_skewness = btc_volatility.skew()
observed_kurtosis = btc_volatility.kurt()

# Print moments
print(f"Mean: {observed_mean}, Variance: {observed_variance}")
print(f"Skewness: {observed_skewness}, Kurtosis: {observed_kurtosis}")


# ### Confirmation of Lognormality

# In[ ]:


from scipy.stats import lognorm, kstest

# Fit a lognormal distribution to the data
shape, loc, scale = lognorm.fit(btc_volatility.dropna(), floc=0)  # `floc=0` forces the location to zero

# Perform the Kolmogorov-Smirnov (K-S) test
d_statistic, p_value = kstest(btc_volatility.dropna(), 'lognorm', args=(shape, loc, scale))
print(f"K-S Statistic: {d_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis: The data does NOT follow a lognormal distribution.")
else:
    print("Fail to reject the null hypothesis: The data may follow a lognormal distribution.")


# ## Mean Vol Before / After

# ### Mann Whitney U Test (no assumed log normality)

# In[ ]:


cycle_high_dates = ['2017-12-17', '2021-11-10']
cycle_low_dates = ['2018-12-15', '2022-11-21']

window = 30


# In[ ]:


from scipy.stats import mannwhitneyu

for high in cycle_high_dates:
    high_date = pd.Timestamp(high)
    before_vol = btc_volatility[high_date - pd.Timedelta(days=window):high_date].sort_values(ascending=True)
    after_vol = btc_volatility[high_date:high_date + pd.Timedelta(days=window)].sort_values(ascending=True)

    print(f"Cycle High: {high}")
    print(f"Avg Volatility Before: {before_vol.mean():.4f}, After: {after_vol.mean():.4f}")
    # Example: Comparing before and after volatilities for a cycle high
    u_stat, p_value = mannwhitneyu(before_vol, after_vol, alternative='two-sided')
    print(f"Mann-Whitney U Test: U-Statistic = {u_stat}, P-Value = {p_value:.4f}")
    
    if p_value < 0.05:
        print("Significant difference between volatilities before and after.")
    else:
        print("No significant difference between volatilities before and after.")

for low in cycle_low_dates:
    low_date = pd.Timestamp(low)
    before_vol = btc_volatility[low_date - pd.Timedelta(days=window):low_date].sort_values(ascending=True)
    after_vol = btc_volatility[low_date:low_date + pd.Timedelta(days=window)].sort_values(ascending=True)

    print(f"Cycle Low: {low}")
    print(f"Avg Volatility Before: {before_vol.mean():.4f}, After: {after_vol.mean():.4f}")
    # Example: Comparing before and after volatilities for a cycle high
    u_stat, p_value = mannwhitneyu(before_vol, after_vol, alternative='two-sided')
    print(f"Mann-Whitney U Test: U-Statistic = {u_stat}, P-Value = {p_value:.4f}")
    
    if p_value < 0.05:
        print("Significant difference between volatilities before and after.")
    
    else:
        print("No significant difference between volatilities before and after.")



# # Statistical Difference in Each Cycles Volatility and Distributions

# In[ ]:


btc_volatility = pd.DataFrame(btc_volatility.dropna())
# Reset the index to turn it into a column
btc_volatility.reset_index(inplace=True)

btc_volatility.head(5)


# ### Separate Cycle's Vols

# In[ ]:


# Define the date ranges for each cycle
cycle1_start = "2016-07-09"
cycle1_end = "2020-05-11"
cycle2_start = "2020-05-11"
cycle2_end = "2024-04-30"

# Create the first cycle DataFrame
btc_cycle1vol = btc_volatility[
    (btc_volatility['Date'] >= cycle1_start) & (btc_volatility['Date'] < cycle1_end)
]

# Create the second cycle DataFrame
btc_cycle2vol = btc_volatility[
    (btc_volatility['Date'] >= cycle2_start) & (btc_volatility['Date'] < cycle2_end)
]

btc_cycle1vol.head(), btc_cycle2vol.head()


# ### Get Each Cycle's Mean Vol

# In[ ]:


btc_cycle1vol_mean = btc_cycle1vol['BTC-USD'].mean()
btc_cycle1vol_mean


# In[ ]:


btc_cycle2vol_mean = btc_cycle2vol['BTC-USD'].mean()
btc_cycle2vol_mean


# ### Confirmation of Signficant Change in Vol Distribution

# In[ ]:


# Perform Mann-Whitney U test
stat, p = mannwhitneyu(btc_cycle1vol['BTC-USD'], btc_cycle2vol['BTC-USD'], alternative='two-sided')

# Display results
print(f"U-statistic: {stat}")
print(f"P-value: {p}")

# Interpret results
if p < 0.05:
    print("The distributions are significantly different.")
else:
    print("The distributions are not significantly different.")


# ### Cycles Vol Distribution vs. Mature Asset (Gold)

# In[ ]:


gold_vol = returns['GC=F'].rolling(window=30).std()
gold_vol.dropna(inplace=True)

SPY_vol = returns['^SPX'].rolling(window=30).std()
SPY_vol.dropna(inplace=True)

halving_date = '2024-04-19'
current_vol = btc_vol['BTC-USD'].loc[btc_vol.index > halving_date].dropna()
current_vol.head()

plt.figure(figsize=(10, 8))

# Plot 1: BTC Volatility Distributions (Cycle 1 and Cycle 2 on the same plot)
plt.subplot(3, 1, 1)  # 2 rows, 1 column, 1st plot
sns.kdeplot(btc_cycle1vol['BTC-USD'], label='Cycle 1', fill=True)
sns.kdeplot(btc_cycle2vol['BTC-USD'], label='Cycle 2', fill=True)
sns.kdeplot(current_vol, label='Cycle 3', fill=True)
sns.kdeplot(current_vol)
plt.legend()
plt.title("BTC Volatility Distributions")
plt.xlabel("BTC-USD Volatility")

# Plot 2: Gold Volatility Distribution on a separate plot
plt.subplot(3, 1, 2)  # 2 rows, 1 column, 2nd plot
sns.kdeplot(gold_vol, label='Gold', fill=True, color='gold')
plt.legend()
plt.title('Gold Volatility Distribution')
plt.xlabel('Gold Volatility')

plt.subplot(3,1,3)
sns.kdeplot(SPY_vol, label = 'SPY', fill = True, color = 'green')
plt.legend()
plt.title('SPY Volatility Distribution')
plt.xlabel('SPY Volatility')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()


# # SARIMA Implementation Upon Vol

# ### Stationarity Check

# In[ ]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(btc_volatility['BTC-USD'].iloc[600:])
print(f"ADF Statistic: {result[0]}")
print(f"P-Value: {result[1]}")
if result[1] < 0.05:
    print("Data is stationary.")
else:
    print("Data is not stationary. Differencing may be needed.")


# #### Differencing If Required

# In[ ]:


# btc_volatility['BTC-USD_diff'] = btc_volatility['BTC-USD'].diff().dropna()


# ### ACF Plot

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf

plot_acf(btc_volatility['BTC-USD'].dropna(), lags=500)
plt.show()


# ### PACF Plot

# In[ ]:


from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(btc_volatility['BTC-USD'].dropna(), lags=50)
plt.show() # results suggest an AR(2) process


# ### Model Fitting - Lower AIC / BIC 

# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define the SARIMA model
# Replace p, d, q (non-seasonal) and P, D, Q, m (seasonal) with values from your analysis
model = SARIMAX(
    btc_volatility['BTC-USD'].iloc[2200:],
    order=(2, 0, 0),  # Non-seasonal order (p, d, q)
    seasonal_order=(1, 1, 0, 12)  # Seasonal order (P, D, Q, m) with m = 12 for monthly seasonality
)

# Fit the SARIMA model
sarima_result = model.fit()

# Print the summary of the model
print(sarima_result.summary())


# ### Forecast x Steps

# In[ ]:


forecast_steps = 51
forecast = sarima_result.forecast(steps=forecast_steps)  # Forecast the next 30 time steps

plt.figure(figsize=(12, 7))
plt.plot(btc_volatility['BTC-USD'], label='Historical Data')
plt.plot(forecast, label='Forecast', color='orange')
plt.legend()
plt.show()


# # Current Vol Environment (Cycle 3)

# In[ ]:


btc_vol.dropna(inplace=True)
btc_vol


# In[ ]:





# In[ ]:


current_vol_mean = current_vol.mean()
current_vol_std = current_vol.std()


# In[ ]:


current_vol_z = (current_vol - current_vol_mean) / current_vol_std


# In[ ]:


# Plot for Cycle 3
plt.figure(figsize=(12, 7))
plt.plot(current_vol.index, current_vol_z, label="Cycle 3 Z-Scores", color="blue")
plt.axhline(y=3, color='red', linestyle='--', label="+3 Std Dev")
plt.axhline(y=2, color='orange', linestyle='--', label="+2 Std Dev")
plt.axhline(y=-2, color='orange', linestyle='--', label="-2 Std Dev")
plt.axhline(y=-3, color='red', linestyle='--', label="-3 Std Dev")
plt.axhline(y=0, color='black', linestyle='-', label="Mean")
plt.title("Bitcoin Cycle 3 Volatility Z-Scores")
plt.xlabel("Time")
plt.ylabel("Z-Score")
plt.legend()
plt.show()


# In[ ]:


diff_current_vol = pd.DataFrame()


# In[ ]:


diff_current_vol = current_vol.diff().dropna()


# In[ ]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(diff_current_vol)
print(f"ADF Statistic: {result[0]}")
print(f"P-Value: {result[1]}")
if result[1] < 0.05:
    print("Data is stationary.")
else:
    print("Data is not stationary. Differencing may be needed.")


# In[ ]:


plot_acf(current_vol.dropna(), lags=30)
plt.show()


# In[ ]:


plot_pacf(current_vol.dropna(), lags=25)
plt.show() # results suggest an AR(2) process


# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define the SARIMA model
# Replace p, d, q (non-seasonal) and P, D, Q, m (seasonal) with values from your analysis
model = SARIMAX(
    current_vol,
    order=(2, 1, 0),  # Non-seasonal order (p, d, q)
    seasonal_order=(0, 0, 1, 2)  # Seasonal order (P, D, Q, m) with m = 12 for monthly seasonality
)

# Fit the SARIMA model
sarima_result = model.fit()

# Print the summary of the model
print(sarima_result.summary())


# In[ ]:


forecast_steps = 51
forecast = sarima_result.forecast(steps=forecast_steps)  # Forecast the next 30 time steps

plt.figure(figsize=(12, 7))
plt.plot(current_vol, label='Historical Data')
plt.plot(forecast, label='Forecast', color='orange')
plt.legend()
plt.show()


# In[ ]:


current_cycle_prices = data['BTC-USD'].loc[data.index > halving_date].dropna()

fig, ax1 = plt.subplots(figsize=(15, 9))

# Plot log-transformed BTC price on the primary y-axis
ax1.plot(current_cycle_prices, color='green', label='BTC Price')
ax1.set_xlabel('Index')

ax1.set_ylabel('BTC Price', color='green')
ax1.tick_params(axis='y', labelcolor='green')

# Create a twin y-axis for BTC volatility
ax2 = ax1.twinx()
ax2.plot(current_vol_z, color='orange', label='BTC Volatility Normalized', linestyle='--')
plt.axhline(y=3, color='red', linestyle='--', label="+3 Std Dev")
plt.axhline(y=2, color='orange', linestyle='--', label="+2 Std Dev")
plt.axhline(y=-2, color='orange', linestyle='--', label="-2 Std Dev")
plt.axhline(y=-3, color='red', linestyle='--', label="-3 Std Dev")
plt.axhline(y=0, color='black', linestyle='-', label="Mean")
ax2.set_ylabel('BTC Volatility Normalized', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Show the plot
plt.show()


# ## Optimising the Rolling Vol Window for Correlation Between the Vol and Returns

# By plotting the rolling window of vol against the returned correlation of said vol values to the returns we can locate (if any) an optimal point where 
# volatility is a good predictor of returns. 

# In[ ]:


rolling_window_k = list(range(2, 201))


# In[ ]:


import pandas as pd

# Initialize an empty list to store volatility columns
volatility_columns = []

# Calculate the rolling volatility for each window and store in the list
current_cycle_returns = current_cycle_prices.pct_change().dropna()

for window in rolling_window_k:
    column_name = f'Volatility_Window_{window}'
    volatility_column = current_cycle_returns.rolling(window).std()
    volatility_columns.append(volatility_column.rename(column_name))

# Concatenate all columns at once to create the final volatility DataFrame
volatility = pd.concat(volatility_columns, axis=1)

# Print the resulting volatility DataFrame
volatility.head()


# In[ ]:


current_cycle_prices_z = (current_cycle_prices - current_cycle_prices.mean()) / current_cycle_prices.std()
current_cycle_prices_z = current_cycle_prices_z.dropna()

# Create an empty list to store the correlation results
correlation_results = []

# Iterate through each column in the volatility DataFrame
for column in volatility.columns:
    # Calculate the rolling window size from the column name (e.g., "Volatility_Window_N")
    window_size = int(column.split("_")[-1])
    
    # Drop the first N (window_size) rows from both the volatility column and returns to align valid data
    valid_volatility = volatility[column].iloc[window_size:]
    valid_z_prices = current_cycle_prices_z.iloc[window_size:]
    
    # Calculate the correlation
    correlation_value = valid_volatility.corr(valid_z_prices)
    
    # Append the result (window size and correlation) to the results list
    correlation_results.append({'Window_Size': window_size, 'Correlation': correlation_value})

# Convert the results list into a DataFrame for analysis
correlation_df = pd.DataFrame(correlation_results)

# Print or save the correlation DataFrame
correlation_df.head()


# In[ ]:


# Find the row with the maximum correlation
max_correlation_row = correlation_df.loc[correlation_df['Correlation'].idxmax()]

# Find the row with the minimum correlation
min_correlation_row = correlation_df.loc[correlation_df['Correlation'].idxmin()]

# Print the results
print("Maximum Correlation:")
print(max_correlation_row)

print("\nMinimum Correlation:")
print(min_correlation_row)


# Now we run the normalized vol and price data again but with the optimised rolling volatility window.

# In[ ]:


optimise_N_vol = current_cycle_returns.rolling(window=200).std().dropna()

optimise_N_vol_mean = optimise_N_vol.mean()
optimise_N_vol_std = optimise_N_vol.std()

optimise_N_vol_z = (optimise_N_vol - optimise_N_vol_mean) / optimise_N_vol_std

plt.plot(optimise_N_vol_z)
plt.axhline(y = optimise_N_vol_mean, color = 'orange', linestyle='--', label = 'Mean')


# In[ ]:


current_cycle_prices_z = (current_cycle_prices - current_cycle_prices.mean()) / current_cycle_prices.std()
current_cycle_prices_z = current_cycle_prices_z.dropna()

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot log-transformed BTC price on the primary y-axis
ax1.plot(current_cycle_prices_z, color='green', label='BTC Price')
ax1.set_xlabel('Index')

ax1.set_ylabel('BTC Price', color='green')
ax1.tick_params(axis='y', labelcolor='green')


# Create a twin y-axis for BTC volatility
ax2 = ax1.twinx()
ax2.plot(optimise_N_vol_z, color='orange', label='BTC Volatility Normalized', linestyle='--')
plt.axhline(y=3, color='red', linestyle='--', label="+3 Std Dev")
plt.axhline(y=2, color='orange', linestyle='--', label="+2 Std Dev")
plt.axhline(y=-2, color='orange', linestyle='--', label="-2 Std Dev")
plt.axhline(y=-3, color='red', linestyle='--', label="-3 Std Dev")
plt.axhline(y=0, color='black', linestyle='-', label="Mean")
ax2.set_ylabel('BTC Volatility Normalized', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Show the plot
plt.show()


# In[ ]:





# # Mean Reversion of BTC Returns in Consecutive Days

# For each cycle, c, (1, 2, 3):
# Collect data about daily returns above a 2 and 3 STD and the consecutive n days return (n = 1, 2, 3, 4 , 5).
# Analyse the rate at which n daily return is + / - and the average return of the aggregate n daily return. 

# In[ ]:


price_data = pd.DataFrame(data['BTC-USD'].dropna())
price_data.head()


# In[ ]:


daily_returns = price_data.pct_change().dropna()
daily_returns.head()


# In[ ]:


daily_returns_mean = daily_returns.mean()
daily_returns_std = daily_returns.std()


# In[ ]:


daily_returns['Z-Score'] = (daily_returns - daily_returns_mean) / daily_returns_std
daily_returns.head()


# In[ ]:


plt.plot(daily_returns['Z-Score'], label='Daily Returns Z')

plt.axhline(y=2, color='r', linestyle='--', label='Standard Deviation +2')
plt.axhline(y=3, color='r', linestyle='-', label='Standard Deviation +3')
plt.axhline(y=-2, color='b', linestyle='--', label='Standard Deviation -2')
plt.axhline(y=-3, color='b', linestyle='-', label='Standard Deviation -3')

plt.title('Daily Returns Z-Score with Standard Deviation Lines')
plt.xlabel('Index')
plt.ylabel('Z-Score')
plt.legend()

plt.show()


# ## Parameters for Aggregate Cycles

# In[ ]:


# days
n = 1

# z-score bands
high_z = 2
low_z = -2

daily_returns = daily_returns.reset_index(drop=True)


# In[ ]:


# List to store results
results = []

# Loop through the dataframe
for i in range(len(daily_returns)):
    if daily_returns.loc[i, 'Z-Score'] > high_z or daily_returns.loc[i, 'Z-Score'] < low_z:  # Trigger condition
        # Calculate cumulative return for the next `n` days
        if i + n < len(daily_returns):
            cumulative_return = daily_returns.loc[i+1:i+n, 'BTC-USD'].sum()
            results.append({
                'trigger_index': i,
                'z_score': daily_returns.loc[i, 'Z-Score'],
                'cumulative_return': cumulative_return
            })

# Convert results into a new dataframe
results_df = pd.DataFrame(results)

# Initialize counters
pos_trigger_count = 0
neg_to_neg_count = 0  # Positive return leading to negative cumulative return

neg_trigger_count = 0
neg_to_pos_count = 0  # Negative return leading to positive cumulative return

# Iterate through the dataframe
for i in range(len(daily_returns)):
    # Case 1: Positive trigger
    if daily_returns.loc[i, 'Z-Score'] > high_z:
        pos_trigger_count += 1
        if i + n < len(daily_returns):
            cumulative_return = daily_returns.loc[i+1:i+n, 'BTC-USD'].sum()
            if cumulative_return < 0:  # Cumulative return is negative
                neg_to_neg_count += 1

    # Case 2: Negative trigger
    if daily_returns.loc[i, 'Z-Score'] < low_z:
        neg_trigger_count += 1
        if i + n < len(daily_returns):
            cumulative_return = daily_returns.loc[i+1:i+n, 'BTC-USD'].sum()
            if cumulative_return > 0:  # Cumulative return is positive
                neg_to_pos_count += 1

# Calculate rates
rate_pos_to_neg = (neg_to_neg_count / pos_trigger_count) * 100 if pos_trigger_count > 0 else 0
rate_neg_to_pos = (neg_to_pos_count / neg_trigger_count) * 100 if neg_trigger_count > 0 else 0



# Display the results
print(results_df)

print(f"Rate of positive z-scores leading to negative returns over {n} days: {rate_pos_to_neg:.2f}%")
\
print(f"Rate of negative z-scores leading to positive returns over {n} days: {rate_neg_to_pos:.2f}%")


# In[ ]:


# Assuming 'results_df' contains 'z_score' and 'cumulative_return' columns
correlation = results_df['z_score'].corr(results_df['cumulative_return'])

print(f"Correlation between Z-Score and Cumulative Return: {correlation}")


# In[ ]:


# +3 STD conversion
std_3 = (high_z * daily_returns_std) + daily_returns_mean
neg_3_std = (low_z * daily_returns_std) + daily_returns_mean


# In[ ]:


std_3, neg_3_std

