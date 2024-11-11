#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# load in data via pandas read csv function and store as dataframe (data)
data = pd.read_csv(r'C:\Users\44734\OneDrive\Desktop\JPM_QR\Task 1\Nat_Gas.csv')
data


# In[ ]:


# create a plot showing the absolute price of natural gas over time
data.plot()
plt.title('Historical Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[ ]:


## Notes ^ 

### Clear there is a drift and also seasonality to the data, where every ~10 months we see a ~10% retrace in prices

### Means data is a submartingale for the period shown, mean may show periodic fluctuations, autocorrelation will be 
### strongly correlated with its previous values & variance may also exhibit periodic changes such as during winter due to 
### increased consumption of natural gas for heating

### 8-12 Jun to Oct | 19-22 May to Sep | 32-34 Jun to Sep | 43-46 May - Sep


# In[ ]:


# import model 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# fit the model
model = ExponentialSmoothing(data['Prices'], trend='add', seasonal='add', seasonal_periods=12).fit()
forecast = model.forecast(steps=12)  # Forecast for an additional 12 months

# plot the forecast
data.plot(label='Historical')
forecast.plot(label='Forecast', linestyle='--')
plt.legend()
plt.show()


# In[ ]:


forecast


# In[ ]:


# turn into pandas dataframe
forecast_df = pd.DataFrame(forecast, columns=['Price'])
forecast_df


# In[ ]:


# adds a date column to forecasted prices dataframe 
forecast_df['Dates'] = pd.date_range(start='2024-09-30', periods=12, freq='M') 
forecast_df.head()


# In[ ]:


def get_price_estimate(date):
    if date in data['Dates']:
        return data.loc[date].values[0]
    elif date in forecast_df.index:
        return forecast_df.loc[date]
    else:
        print('Date out of range for available forecast')


# In[ ]:


# please use the index number to the corresponding date
print(get_price_estimate(58))


# In[ ]:




