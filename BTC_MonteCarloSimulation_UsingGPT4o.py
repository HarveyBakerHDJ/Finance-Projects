#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import statsmodels as st
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


# In[ ]:


# Define the ticker symbol for Bitcoin futures
# Note: The ticker for Bitcoin futures on Yahoo Finance might vary, here's a common one: BTC-USD
ticker_symbol = 'BTC-USD'

# Fetch historical data for Bitcoin futures
bitcoin_futures_data = pd.DataFrame(yf.download(ticker_symbol, start='2020-01-01', end='2024-12-31'))

# Display the fetched data
bitcoin_futures_data.tail()


# In[ ]:


bitcoin_futures_data['Returns'] = bitcoin_futures_data['Close'].pct_change()
bitcoin_futures_data = bitcoin_futures_data.dropna()
bitcoin_futures_data.head()


# In[ ]:


initial_price = bitcoin_futures_data['Close'][-1]  # Use the last available closing price
expected_return = bitcoin_futures_data['Returns'].mean()  # Mean of daily returns
volatility = bitcoin_futures_data['Returns'].std()  # Standard deviation of daily returns
time_horizon = 1  # Time horizon in years
steps = 252  # Number of steps (trading days in a year)
simulations = 10000  # Number of simulation paths
dt = time_horizon / steps


# In[ ]:


random_samples = np.random.normal(0, 1, (steps, simulations))


# In[ ]:


prices = np.zeros((steps, simulations))
prices[0] = initial_price

for t in range(1, steps):
    prices[t] = prices[t-1] * np.exp((expected_return - 0.5 * volatility**2) * dt +
                                      volatility * np.sqrt(dt) * random_samples[t])


# In[ ]:


# Final prices after the simulation
final_prices = prices[-1]
mean_final_price = np.mean(final_prices)
probability_above_threshold = np.mean(final_prices > 60000)

print(f"Expected Final Price: {mean_final_price}")
print(f"Probability of Price Exceeding $60,000: {probability_above_threshold * 100:.2f}%")

# Plot the simulation results
plt.figure(figsize=(10, 6))
plt.plot(prices[:, :100], lw=1)  # Plotting the first 100 simulation paths for clarity
plt.title('Monte Carlo Simulation of BTC Futures Prices')
plt.xlabel('Time (days)')
plt.ylabel('BTC Futures Price')
plt.show()


# In[ ]:




