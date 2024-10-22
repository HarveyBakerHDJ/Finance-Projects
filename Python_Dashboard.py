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

# import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import datetime as dt
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# ## Main:

# #### This section contains the code for both the Dash application and also the backend data handling & modeling

# +
# // intitialize Dash application
app = dash.Dash(__name__, external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css'
])
# //// creating the application layout and divisions with styling
# // creates a division element to contain child elements 
app.layout = html.Div([
    html.Div([
        # // inside the main container
        # creates a heading (h1) 
        html.H1("Stock Visualization Dashboard", className="text-3xl font-bold mb-4 text-gray-800"),
        # // creates an inner division section
        # contains input fields and a 'submit' button
        html.Div([
            # creates input field with specfic styling
            dcc.Input(
                id="ticker-input",
                type="text",
                placeholder="Enter a stock ticker",
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            ),
            # creates a 'submit' button 
            html.Button(
                "Submit",
                id="submit-button",
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline ml-2"
            )
        ], className="flex items-center mb-4"),
        # creates a division for displaying error messages, styled red
        html.Div(id="error-message", className="text-red-500 mb-4")
    # main container is centered on the page with margins & padding
    ], className="container mx-auto px-4 py-8"),

    # this division is a flex container with a negative horizontal margin and bottom margin
    html.Div([
        # creates a division containing a candlestick graph with specific sizing and layout
        html.Div([
            dcc.Graph(id="candlestick-graph", className="h-96")
        ], className="w-full lg:w-1/2 p-2"),
        # creates a division to house the volume graph
        html.Div([
            dcc.Graph(id="volume-graph", className="h-96")
        ], className="w-full lg:w-1/2 p-2")
    ], className="flex flex-wrap -mx-2 mb-4"),

    # this division houses the returns graph
    html.Div([
        dcc.Graph(id="returns-graph", className="h-96")
    ], className="w-full p-2"),

    # // loading component
    # creates a placeholder division to display a 'loading spinner' while content loads 
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="loading-output")
    )
# creates a main container that fills the screen with grey background
], className="bg-gray-100 min-h-screen")

# //// Updating components decorator
# // callback function to update the components based on user interactions

@app.callback(
    [Output("candlestick-graph", "figure"),
     Output("volume-graph", "figure"),
     Output("returns-graph", "figure"),
     Output("loading-output", "children"),
     Output("error-message", "children")],
    Input("submit-button", "n_clicks"), # submit button is our user interaction to update based on 
    State("ticker-input", "value") # captures the value entered in the ticker input field
)

## //// creating an function to update graphs
## defines a function update_graphs to take two arguments n_clicks & ticker
def update_graphs(n_clicks, ticker):
    # checks if the ticker is an empty string or none
    if ticker is None or ticker.strip() == "":
        # if so, returns 3 empty figures & an error message
        return [go.Figure()] * 3 + [None, "Please enter a ticker symbol."]
    # // try block for data download
    try:
        # tries to download stock data for given ticker using yfinance
        df = yf.download(ticker, start="2020-01-01", end="2023-12-31")
        # // handling empty data
        # checks if data is empty
        if df.empty:
            # if so, raises a ValueError indicating no data was found
            raise ValueError("No data found for the given ticker.")

        # // candlestick chart
        # intializes a candlestick chart, sets the x axis to the dates, and the open/high/close/low to respective data points
        candlestick = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
        ## // update candlestick chart
        # updates the layout of the candlestick chart with specific styling - title, axis title, template and margins 
        candlestick.update_layout(
            title=f"{ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        
        
        
        # // rolling beta chart
        # downnloads the s&p500 data
        sp500 = yf.download('^GSPC', start="2020-01-01", end="2023-12-31")
        
        # // calculate daily returns
        stock_returns = df['Close'].pct_change().dropna() # uses .pct_change on both s&p and inputted ticker
        sp500_returns = sp500['Close'].pct_change().dropna() # also drops NaN values
        
        # // ensure indices match
        # returns the intersection of index labels from both dataframes to return common dates
        common_dates = stock_returns.index.intersection(sp500_returns.index)
        # filters out uncommon dates to only include rows with common dates
        stock_returns = stock_returns.loc[common_dates]
        sp500_returns = sp500_returns.loc[common_dates]

        # // calculate rolling beta
        # calculates the rolling co-variance over 30 days 
        rolling_cov = stock_returns.rolling(window=30).cov(sp500_returns)
        # calculates the rolling variance of both over 30 day windows
        rolling_var = sp500_returns.rolling(window=30).var()
        # calculates the rolling beta
        rolling_beta = rolling_cov / rolling_var

        # //// create rolling beta chart
        # initializes a figure
        rolling_beta_fig = go.Figure()

        # // adds a line plot to figure
        rolling_beta_fig.add_trace(go.Scatter(
            # sets x-axis to index(dates) of rolling beta 
            x=rolling_beta.index,
            # sets y-axis to values of rolling beta returned previosuly ^
            y=rolling_beta,
            # display as line
            mode='lines',
            # styles line to blue and 2px wide
            line=dict(color='blue', width=2),
            # initializes title
            name=f'{30}-Day Rolling Beta'
        ))

        # // creates horizontal line to figure
        rolling_beta_fig.add_shape(
            # specifies shape
            type="line",
            # sets co-ordinates to span entire x-axis range
            x0=rolling_beta.index[0],
            # sets y co-ords to 1
            y0=1,
            # # sets co-ordinates to span entire x-axis range
            x1=rolling_beta.index[-1],
            # sets y co-ords to 1
            y1=1,
            # styles line to red, 2px wide and dash style
            line=dict(color="red", width=2, dash="dash"),
        )

        # // updating the layout of the figure
        rolling_beta_fig.update_layout(
            # sets title
            title=f'{ticker} Rolling Beta to S&P 500',
            # labels x-axis
            xaxis_title='Date',
            # labels y-axis
            yaxis_title='Beta',
            # applies white background template
            template='plotly_white',
            # ensures legend is displayed
            showlegend=True
        )

         
        
        # //// mean daily returns chart
        # adds 'returns' column to 'df' 
        # pct_change method calculates change from close price to the next close price
        # essentially calculates the daily return
        df['Returns'] = df['Close'].pct_change()

        # // calculate the mean daily return for each stock
        mean_daily_returns = df['Returns'].mean()

        # // creates figure for mean daily returns
        # creates new figure, scatter graph, sets x & y axis to df[column]
        mean_daily_returns = go.Figure(data=[go.Scatter(x=df.index, y=df['Returns'], mode='lines')])
        # specifies styling of chart
        mean_daily_returns.update_layout(
            # title
            title=f"Mean {ticker} Daily Returns",
            # x axis
            xaxis_title="Date",
            # y axis
            yaxis_title="Returns",
            # template
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        # returns our charts, None & empty string which are both for error messages
        return candlestick, rolling_beta_fig, mean_daily_returns, None, ""

    
    
    # // exception handling
    # catches any exception within the 'try' block
    except Exception as e:
        # creates a user friendly error message, including the error type 
        error_message = f"An error occurred: {str(e)}. Please check the ticker symbol and try again."
        # returns the 3 empty charts and error message
        return [go.Figure()] * 3 + [None, error_message]

if __name__ == "__main__":
    # starts Dash server with debugging enabled
    app.run_server(debug=True)
# -


