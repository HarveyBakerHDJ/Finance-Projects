# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## CMD prompt to run application in streamlit

# +
# streamlit run C:\your\directory\path\black_scholes_model.py
# -

# ## Library Imports

import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


# ## Python Application 

# #### Black-Scholes Pricing Model

# / calculation of theoretical option price block
def calculate_black_scholes(S, K, T, r, sigma, option_type='call'): # defines our black scholes function and the arguments taken
    """Calculate Black-Scholes option price and Greeks"""
    
    # / calculates d1 and d2
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T)) 
    d2 = d1 - sigma*np.sqrt(T)
    
    # // calculates option price
    # / block for call options
    if option_type == 'call': # checks if option_type is call
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) # calculates price specific to put
        delta = norm.cdf(d1) # calculates delta specific to put
        theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2) # calculates theta specific to put

    # / block for put options
    else:  
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1) # calculates price specific to put
        delta = norm.cdf(d1) - 1 # calculates delta specific to put
        theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2) # calculates theta specific to put
    
    # / block for greeks that are the same for both calls and puts
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T)) # calculates gamma 
    vega = S*np.sqrt(T)*norm.pdf(d1) # calculates vega
    rho = K*T*np.exp(-r*T)*norm.cdf(d2) if option_type == 'call' else -K*T*np.exp(-r*T)*norm.cdf(-d2) # calculates rho 

    # / returns values to a dictionary
    return {
        'price': price, 
        'delta': delta, # delta measures option prices sensitivity to changes in the underlying asset's price
        'gamma': gamma, # gamma measures the rate of change of delta with respect to underlying's price
        'theta': theta, # time decay, rate of decline in option price as expiration approaches
        'vega': vega, # vega measures sensitivity of option price to changes of implied volatility in the underlying
        'rho': rho # rho measures the sensitivity of the options price to changes in risk free rate 
    }


# #### Creating Plot

# / create sensitivity line plot block
def create_sensitivity_plot(param_range, prices, param_name): # defines function and the arguments taken
    """Create a plotly figure for sensitivity analysis"""

    # / data housing
    df = pd.DataFrame({ # creates pandas dataframe to house params
        param_name: param_range, # column for param_name and param_values
        'Option Price': prices # column for corresponding option prices
    })
    
    # / creating plot
    fig = px.line(df, x=param_name, y='Option Price', # creates a plotly line plot with specifications
                  title=f'Option Price Sensitivity to {param_name}') # sets title

    # / customises plot layout
    fig.update_layout( 
        xaxis_title=param_name, # x-axis label
        yaxis_title='Option Price', # y-axis label
        showlegend=False # makes sure legend is shown
    )
    return fig # returns fig for display / further use 


# ## Streamlit Application

# + editable=true slideshow={"slide_type": ""}
# / initialize main for function execution
def main():
    st.set_page_config(page_title="Black-Scholes Option Calculator", layout="wide") # configure page title  in browser and layout
    
    st.title("Black-Scholes Option Calculator") # sets header of web app 
    st.markdown("""
    Calculate European-style option prices and analyze sensitivities using the Black-Scholes model.
    """) # subheading explaining purpose
    
    # / create two columns for input parameters
    col1, col2 = st.columns(2) 

    # / column 1 block
    with col1: # creates context for widgets 
        st.subheader("Option Parameters") # creates subheading for column
        option_type = st.selectbox("Option Type", ['call', 'put']) # dropdown selection box for selecting option type
        S = st.number_input("Stock Price (S)", min_value=0.01, value=100.0, step=1.0) # number field input for stock price
        K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0) # number field input for strike price
        
        # / time input block
        st.write("Time to Expiry") # adds text label to describe purpose
        today = datetime.today() # intializes variable as todays date
        expiry_date = st.date_input( # creates date input field for expiry date
            "Expiry Date",
            min_value=today, # ensures min value to enter is 'today' 
            value=today + timedelta(days=365) # sets default value as one year from 'today'
        )

        # / time to expiry block
        T = (expiry_date - today.date()).days / 365.0 # calculates the time to expiry 

    # column 2 block 
    with col2: # creates context for widgets
        st.subheader("Market Parameters") # adds subheader for market parameters
        r = st.number_input("Risk-free Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f") # input field for risk free rate to 3 decimal place format, increment steps, default value and max/min values
        sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.2, step=0.01, format="%.2f") # input field for volatility to 2 decimal place format, increment steps, default value and max/min values 
    
    # / calculate results try block, if there is an error this is 'caught' by the except block
    try:
        results = calculate_black_scholes(S, K, T, r, sigma, option_type) # calls black-scholes function onto the inputted parametrs and stores in results variable
        
        # / block for layout of results
        st.subheader("Results") # adds a subheader for the results 
        col1, col2, col3 = st.columns(3) # creates 3 neat columns 

        # / column 1 block
        with col1: # create context for widgets
            st.metric("Option Price", f"${results['price']:.2f}") # displays the option price with 2 decimal place and a $
            st.metric("Delta", f"{results['delta']:.4f}") # displays the delta formatted to decimal places

        # / column 2 block
        with col2: # create context for widgets
            st.metric("Gamma", f"{results['gamma']:.4f}") # displays gamme to 4 d.p
            st.metric("Theta", f"{results['theta']:.4f}") # displays theta ti 4 d.p

        # / column 3 block
        with col3: # create context for widgets
            st.metric("Vega", f"{results['vega']:.4f}") # displays vega to 4 d.p
            st.metric("Rho", f"{results['rho']:.4f}") # displays rho to 4 d.p
        
        # / sensitivity Analysis block
        st.subheader("Sensitivity Analysis") # creates new subheader for this section
        
        # / prepare data for plots block
        base_params = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
        
        # / create tabs for different sensitivities block
        tab1, tab2, tab3, tab4 = st.tabs(["Stock Price", "Volatility", "Time", "Interest Rate"]) # creates 4 tabs with corresponding titles

        # / stock range to price tab
        with tab1:
            stock_range = np.linspace(max(0.1, S-50), S+50, 100) # creates 100 stock price values, dynamically using the inputted params
            prices = [calculate_black_scholes(s, K, T, r, sigma, option_type)['price'] for s in stock_range] # calculates opton price for each stock price value and appends to a list
            st.plotly_chart(create_sensitivity_plot(stock_range, prices, 'Stock Price')) # creates sensitivity plot by calling function upon arguments

        # / volatility to price tab
        with tab2:
            vol_range = np.linspace(0.05, 1.0, 100) # intialises a volatility value range (of 100 values) from 0.05 to 100 as a numpy array
            prices = [calculate_black_scholes(S, K, T, r, v, option_type)['price'] for v in vol_range] # appends prices to a list after calculating option price for each value in volatility range
            st.plotly_chart(create_sensitivity_plot(vol_range, prices, 'Volatility')) # displays plot by calling the create plot function onto volatility range and prices, with volatility as the x axis label

        # / time to expiry to price tab
        with tab3: 
            time_range = np.linspace(0.01, 2.0, 100) # creates time to expiry range
            prices = [calculate_black_scholes(S, K, t, r, sigma, option_type)['price'] for t in time_range] # ""
            st.plotly_chart(create_sensitivity_plot(time_range, prices, 'Time to Expiry')) # "" 

        # / risk free rate to price tab
        with tab4:
            rate_range = np.linspace(0.0, 0.2, 100) # creates risk free rate range 
            prices = [calculate_black_scholes(S, K, T, rt, sigma, option_type)['price'] for rt in rate_range] # ""
            st.plotly_chart(create_sensitivity_plot(rate_range, prices, 'Interest Rate')) # ""
    
    # / error catching block         
    except Exception as e: 
        st.error(f"An error occurred: {str(e)}") # displays error message within streamlit app
        st.info("Please check your input parameters and try again.") # advises users to review their inputs to correct any issues

if __name__ == '__main__':
    main()
# -


