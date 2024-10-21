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

import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


def calculate_black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price and Greeks"""
    
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Calculate option price
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put option
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    
    # Greeks that are the same for both calls and puts
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*np.sqrt(T)*norm.pdf(d1)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2) if option_type == 'call' else -K*T*np.exp(-r*T)*norm.cdf(-d2)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


def create_sensitivity_plot(param_range, prices, param_name):
    """Create a plotly figure for sensitivity analysis"""
    df = pd.DataFrame({
        param_name: param_range,
        'Option Price': prices
    })
    
    fig = px.line(df, x=param_name, y='Option Price',
                  title=f'Option Price Sensitivity to {param_name}')
    fig.update_layout(
        xaxis_title=param_name,
        yaxis_title='Option Price',
        showlegend=False
    )
    return fig


# +
def main():
    st.set_page_config(page_title="Black-Scholes Option Calculator", layout="wide")
    
    st.title("Black-Scholes Option Calculator")
    st.markdown("""
    Calculate European-style option prices and analyze sensitivities using the Black-Scholes model.
    """)
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option Parameters")
        option_type = st.selectbox("Option Type", ['call', 'put'])
        S = st.number_input("Stock Price (S)", min_value=0.01, value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
        
        # Time input using date picker
        st.write("Time to Expiry")
        today = datetime.today()
        expiry_date = st.date_input(
            "Expiry Date",
            min_value=today,
            value=today + timedelta(days=365)
        )
        T = (expiry_date - today.date()).days / 365.0
        
    with col2:
        st.subheader("Market Parameters")
        r = st.number_input("Risk-free Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f")
        sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.2, step=0.01, format="%.2f")
    
    # Calculate results
    try:
        results = calculate_black_scholes(S, K, T, r, sigma, option_type)
        
        # Display results
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Option Price", f"${results['price']:.2f}")
            st.metric("Delta", f"{results['delta']:.4f}")
        
        with col2:
            st.metric("Gamma", f"{results['gamma']:.4f}")
            st.metric("Theta", f"{results['theta']:.4f}")
        
        with col3:
            st.metric("Vega", f"{results['vega']:.4f}")
            st.metric("Rho", f"{results['rho']:.4f}")
        
        # Sensitivity Analysis
        st.subheader("Sensitivity Analysis")
        
        # Prepare data for sensitivity plots
        base_params = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
        
        # Create tabs for different sensitivities
        tab1, tab2, tab3, tab4 = st.tabs(["Stock Price", "Volatility", "Time", "Interest Rate"])
        
        with tab1:
            stock_range = np.linspace(max(0.1, S-50), S+50, 100)
            prices = [calculate_black_scholes(s, K, T, r, sigma, option_type)['price'] for s in stock_range]
            st.plotly_chart(create_sensitivity_plot(stock_range, prices, 'Stock Price'))
        
        with tab2:
            vol_range = np.linspace(0.05, 1.0, 100)
            prices = [calculate_black_scholes(S, K, T, r, v, option_type)['price'] for v in vol_range]
            st.plotly_chart(create_sensitivity_plot(vol_range, prices, 'Volatility'))
        
        with tab3:
            time_range = np.linspace(0.01, 2.0, 100)
            prices = [calculate_black_scholes(S, K, t, r, sigma, option_type)['price'] for t in time_range]
            st.plotly_chart(create_sensitivity_plot(time_range, prices, 'Time to Expiry'))
        
        with tab4:
            rate_range = np.linspace(0.0, 0.2, 100)
            prices = [calculate_black_scholes(S, K, T, rt, sigma, option_type)['price'] for rt in rate_range]
            st.plotly_chart(create_sensitivity_plot(rate_range, prices, 'Interest Rate'))
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your input parameters and try again.")

if __name__ == '__main__':
    main()
# -


