#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import random as rd


# ### Parameters

# In[ ]:


S = 100
mu = 0.095
sigma = 0.175
T = 1 
N = 2000
simulations = 1

dt = T / N


# ### Simulation

# In[ ]:


np.random.seed(42) # sets a starting point for the random numbers 
S_t = np.zeros((N + 1 ,simulations)) # creates an array filled with 0s, has (N + 1) rows and (simulations) columns
innovations = np.zeros(N)
S_t[0] = S # sets the first row of the array to the intial stock price (s)


# In[ ]:


for t in range(1, N + 1): # starts the loop from 1 to N 
    Z = np.random.standard_normal() # generates (simulations) number of random numbers from a stan norm dist (mean0, std 1)
    innovations[t-1]=Z # store each innovation
    S_t[t] = S_t[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * Z * np.sqrt(dt)) # GBE/Ito equation 


# ### Plotting

# In[ ]:


# Plot the stock price
plt.figure(figsize=(10, 6))
plt.plot(S_t, label='Stock Price Path')
plt.title('Simulated Stock Price using Geometric Brownian Motion')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plot the innovations (random shocks)
plt.figure(figsize=(10, 6))
plt.plot(innovations, label='Innovations')
plt.title('Innovations in Stock Price Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Innovation Value')
plt.legend()
plt.show()


# In[ ]:




