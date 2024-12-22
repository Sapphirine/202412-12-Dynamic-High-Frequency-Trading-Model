#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:50:31 2024

@author: hhhao1004
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas_market_calendars as mcal
import cvxpy as cp
import pendulum
import json
import plotly
import plotly.graph_objs as go

# Define the PredictionDataset class
# Custom dataset class for handling input data in PyTorch
class PredictionDataset(Dataset):
    def __init__(self, x):
        # Initialize the dataset with input data, converting each element to a PyTorch tensor of type float
        self.x = list(map(lambda x: torch.tensor(x).float(), x))

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve a single sample from the dataset by index
        return self.x[idx]

# Define the GRU model
# Custom GRU-based neural network model for sequence prediction tasks
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        # Initialize GRU model parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer for processing sequential input data
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer for mapping GRU outputs to target size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize the hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through the GRU layer
        out, _ = self.gru(x, h0)

        # Pass the last hidden state output through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# GRUTrainer class
# Trainer class for handling GRU model training and prediction
class GRUTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        # Initialize the trainer with model, loss function, optimizer, scheduler, and device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Move the model to the specified device
        self.model.to(self.device)

    def predict(self, dataloader):
        # Make predictions using the trained model
        self.model.eval()  # Set the model to evaluation mode
        predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                # Move inputs to the specified device
                inputs = inputs.to(self.device)

                # Forward pass through the model to get predictions
                outputs = self.model(inputs)

                # Collect predictions in CPU memory for further processing
                predictions.append(outputs.cpu().numpy())
        # Combine all predictions into a single array
        return np.concatenate(predictions, axis=0)

# Define the loss function
# Custom loss function to calculate the negative Information Coefficient (IC)
class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Flatten predictions and true values to 1D
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Calculate the mean of predictions and true values
        y_pred_mean = torch.mean(y_pred)
        y_true_mean = torch.mean(y_true)

        # Calculate the numerator and denominator for the IC formula
        numerator = torch.sum((y_pred - y_pred_mean) * (y_true - y_true_mean))
        denominator = torch.sqrt(torch.sum((y_pred - y_pred_mean) ** 2) * torch.sum((y_true - y_true_mean) ** 2))

        # Compute IC and return its negative for loss optimization
        ic = numerator / (denominator + 1e-8)  # Added epsilon to prevent division by zero
        return -ic  # Negative IC as loss to maximize IC during training

# Function to check if today is a trading day
# Check whether the current date is a valid trading day on the NYSE
def check_trading_day(**kwargs):
    today_date = (datetime.today()).strftime('%Y-%m-%d')  # Get the current date as a string
    nyse = mcal.get_calendar('NYSE')  # Get the NYSE trading calendar
    schedule = nyse.valid_days(start_date=today_date, end_date=today_date)  # Check if today is a trading day
    return not schedule.empty  # Return True if today is a trading day, False otherwise

# Main function for DAG
def calculate_weight(**kwargs):
    
    # Function to calculate the last Friday before the current date
    def calculate_last_friday(date):
        offset = (date.weekday() - 4) % 7  # Calculate offset to last Friday
        if date.weekday() <= 5:  # Adjust for days earlier than Friday this week
            offset += 7
        return date - timedelta(days=offset)
    
    # Find the last Friday and format it as a string
    last_fri = calculate_last_friday(datetime.today()).strftime('%Y-%m-%d')
    print(f'Using {last_fri} data...')

    # Define the tickers and date range for fetching data
    tickers = [
        "META", "TSLA", "AMZN", "GOOG", "AAPL", "MSFT", "NVDA", "MRNA", "SMCI",
        "ILMN", "MDB", "BIIB", "CDW", "GFS", "WBD", "ON", "ANSS", "DXCM", "ZS",
        "CSGP", "TTWO", "IDXX", "MCHP", "CCEP", "KHC", "CTSH", "EXC", "LULU",
        "VRSK", "XEL", "BKR", "EA", "KDP", "ODFL", "DDOG", "FAST", "ROST", "AEP",
        "PAYX", "FANG", "MNST", "CPRT", "CHTR", "NXPI", "PCAR", "ROP", "TTD", "TEAM",
        "ADSK", "CSX", "ORLY", "WDAY", "FTNT", "DASH", "MAR", "MRVL", "REGN",
        "KLAC", "CDNS", "SNPS", "PYPL", "ABNB", "CRWD", "MDLZ", "CTAS", "LRCX", "MELI",
        "INTC", "ADI", "APP", "MU", "GILD", "SBUX", "VRTX", "ADP", "PANW", "SIRI",
        "AMAT", "HON", "AMGN", "PDD", "CMCSA", "BKNG", "QCOM", "TXN", "INTU", "ISRG",
        "AZN", "LIN", "PEP", "ADBE", "AMD", "CSCO", "ASML", "TMUS", "NFLX", "COST",
        "AVGO", "MSTR", "PLTR"
    ]
    end_date = (datetime.strptime(last_fri, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=140)).strftime("%Y-%m-%d")
    
    # Fetch historical stock data for the tickers
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d", group_by="ticker")
    
    # Function to calculate various financial factors for analysis
    def calculate_factors(df, market_df=None):
        factors = pd.DataFrame(index=df.index)
    
        # Calculate moving averages
        factors['SMA_10'] = df['Close'].rolling(window=10).mean()
        factors['SMA_30'] = df['Close'].rolling(window=30).mean()
    
        # Calculate exponential moving averages
        factors['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        factors['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    
        # Bollinger Bands
        rolling_std = df['Close'].rolling(window=10).std()
        factors['BB_Upper'] = factors['SMA_10'] + 2 * rolling_std
        factors['BB_Lower'] = factors['SMA_10'] - 2 * rolling_std
    
        # Momentum calculation
        factors['Momentum'] = df['Close'] - df['Close'].shift(10)
    
        # Calculate beta if market data is provided
        if market_df is not None:
            market_returns = market_df['Close'].pct_change()
            stock_returns = df['Close'].pct_change()
            factors['Beta'] = (
                stock_returns.rolling(window=30).cov(market_returns) /
                market_returns.rolling(window=30).var()
            )
    
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        factors['RSI'] = 100 - (100 / (1 + rs))
    
        # Stochastic Oscillator
        lowest_low = df['Low'].rolling(window=14).min()
        highest_high = df['High'].rolling(window=14).max()
        factors['Stochastic'] = (
            (df['Close'] - lowest_low) / (highest_high - lowest_low)
        ) * 100
    
        # Money Flow Index (MFI)
        if 'Volume' in df.columns:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_flow = money_flow.where(delta > 0, 0).rolling(window=14).sum()
            negative_flow = money_flow.where(delta < 0, 0).rolling(window=14).sum()
            factors['MFI'] = 100 - (100 / (1 + (positive_flow / negative_flow)))
    
        # Include additional features for analysis
        factors['Open'] = df['Open']
        factors['High'] = df['High']
        factors['Low'] = df['Low']
        factors['Adj Close'] = df['Adj Close']
        factors['Volume'] = df['Volume']
    
        return factors
    
    # Dictionary to store calculated factors for each ticker
    factors_dict = {}
    for ticker in tickers:
        try:
            # Calculate factors for each stock and take the last 60 rows
            stock_data = data[ticker]
            factors_dict[ticker] = calculate_factors(stock_data).iloc[-60:]
        except KeyError:
            print(f"Data for {ticker} not found. Skipping.")
    
    # Combine all calculated factors into a single DataFrame
    all_factors = pd.concat(factors_dict, axis=1, keys=factors_dict.keys())
    
    # Normalize factors for each stock
    factors = all_factors.columns.get_level_values(1).unique()
    normalized_factors = {}
    for factor in factors:
        # Perform normalization for the given factor
        factor_df = all_factors.xs(factor, level=1, axis=1)
        factor_mean_values = factor_df.mean(axis=1)
        factor_std_values = factor_df.std(axis=1)
        factor_df_normalized = (factor_df.T - factor_mean_values).T / factor_std_values.values[:, None]
        normalized_factors[factor] = factor_df_normalized
    
    # Prepare data for GRU model prediction
    window_size = 60
    merged_tickers = {ticker: pd.DataFrame() for ticker in tickers}
    rolling_list = []
    
    # Merge normalized factors for each ticker
    for factor, df in normalized_factors.items():
        for ticker in tickers:
            merged_tickers[ticker][factor] = df[ticker]
    
    # Create rolling windows for GRU model input
    for ticker, merged_df in merged_tickers.items():
        data = merged_df.to_numpy()
        shape = (data.shape[0] - window_size + 1, window_size, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        rolling_windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        rolling_list.append(rolling_windows)

    # Concatenate rolling windows for all tickers
    rolling_list = np.concatenate(rolling_list)

    # Load the pre-trained GRU model
    file_path = os.path.expanduser("~/airflow/dags/gru.pth")
    model = GRUModel(15, 64, 2, 1)  # Initialize GRU model with input, hidden, and output sizes
    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu'), weights_only=True))  # Load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, otherwise CPU
    model.to(device)  # Move model to the specified device
    tester = GRUTrainer(model, ICLoss(), None, None, device)  # Initialize GRU trainer with the model

    # Make predictions using the GRU model
    prediction_dataset = PredictionDataset(rolling_list)  # Prepare dataset for predictions
    prediction_loader = DataLoader(prediction_dataset, batch_size=100, shuffle=False)  # Create DataLoader for batch predictions
    predictions = tester.predict(prediction_loader).flatten()  # Generate predictions and flatten the results
    ranks = pd.qcut(predictions, 10, labels=False) + 1  # Rank predictions into deciles (1-10)
    ranked_tickers = pd.DataFrame({"Ticker": tickers, "Rank": ranks, "Predicted Return": predictions})  # Create DataFrame for ranked tickers
    print(ranked_tickers)  # Display ranked tickers and predicted returns

    # Fetch historical data to calculate weekly returns
    data = yf.download(tickers, start=start_date, end=end_date, progress=True)
    num_day_return = 5  # Define the number of days for return calculation

    # Validate the fetched data
    if data.empty:
        raise ValueError("No data fetched. Please check the tickers and date range.")
    if len(tickers) == 1:
        close_prices = data['Close'].to_frame()  # Single ticker case: wrap Close prices into a DataFrame
        open_prices = data['Open'].to_frame()  # Single ticker case: wrap Open prices into a DataFrame
    else:
        close_prices = data['Close']
        open_prices = data['Open']
    if close_prices.isnull().any().any():
        raise ValueError("close_prices contains NaN values. Please check the data.")
    if open_prices.isnull().any().any():
        raise ValueError("open_prices contains NaN values. Please check the data.")

    # Calculate weekly returns as percentages
    print("Calculating weekly returns...")
    weekly_returns = (close_prices.shift(-num_day_return) - open_prices.shift(-1)) / open_prices.shift(-1)
    weekly_returns_percentage = weekly_returns * 100  # Convert to percentage format
    weekly_returns_percentage.dropna(how='any', inplace=True)  # Drop rows with missing data
    historical_dates = weekly_returns_percentage.iloc[-60:].index  # Get the last 60 dates for historical returns

    # Extract tickers in the top rank (rank 10)
    rank10_tickers = ranked_tickers[ranked_tickers['Rank'] == 10].sort_values(by='Predicted Return', ascending=False)
    long_stocks = rank10_tickers['Ticker'].tolist()  # List of tickers to invest in
    historical_returns = weekly_returns_percentage.loc[historical_dates, long_stocks].dropna()  # Historical returns for top tickers
    expected_returns = historical_returns.mean()  # Calculate expected returns
    cov_matrix = historical_returns.cov()  # Calculate covariance matrix of returns

    # Define portfolio optimization variables
    w = cp.Variable(len(long_stocks))  # Variable for portfolio weights

    # Portfolio return and variance
    portfolio_return = expected_returns.values @ w  # Portfolio return
    portfolio_variance = cp.quad_form(w, cov_matrix.values)  # Portfolio variance

    # Objective function: maximize a Sharpe-like metric (return - lambda * variance)
    lambda_reg = 0.01  # Regularization parameter for variance penalty
    objective = cp.Maximize(portfolio_return - lambda_reg * portfolio_variance)

    # Portfolio constraints
    constraints = [
        cp.sum(w) == 1,  # Sum of weights must equal 1 (fully invested)
        w >= 0,  # No short selling (long-only portfolio)
        w <= 0.2  # Limit weight of any single stock to 20%
    ]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    print("\nSolving the optimization problem with weight constraints...")
    prob.solve()

    # Extract optimized portfolio weights
    if prob.status not in ["infeasible", "unbounded"]:
        optimized_weights = w.value
    else:
        optimized_weights = np.array([0.1] * 10)  # Default weights if optimization fails

    # Fetch daily data for the current date to calculate returns
    today_date = datetime.strptime(pd.Timestamp.now(tz='America/New_York').strftime("%Y-%m-%d"), "%Y-%m-%d")
    data_daily = yf.download(tickers, start=today_date, end=today_date, interval="1d", group_by="ticker")
    target_stocks = data_daily.iloc[-1].loc[long_stocks]  # Extract data for target tickers
    target_returns = (target_stocks.loc[:, 'Close'] - target_stocks.loc[:, 'Open']) / target_stocks.loc[:, 'Open']  # Calculate daily returns

    # Create portfolio DataFrame
    portfolio = pd.DataFrame({
        'Ticker': long_stocks,
        'Weight': optimized_weights,
        'Open': target_stocks.loc[:, 'Open'],
        'Close': target_stocks.loc[:, 'Close'],
        'Daily Return in Percentage': target_returns
    })
    portfolio['Daily Position'] = round(portfolio['Weight'] * portfolio['Daily Return in Percentage'] * 100000, 2)  # Calculate position values

    # Define initial cash and file paths for saving portfolio data
    INITIAL_CASH = 100000
    portfolio_values_path = os.path.expanduser("~/airflow/dags/portfolio_values.csv")
    portfolio_positions_path = os.path.expanduser("~/airflow/dags/portfolio_positions.csv")
    optimized_tickers = long_stocks

    # Function to find the first trading day of the week
    def find_trading_day(date):
        today = date  # Use the provided date
        nyse = mcal.get_calendar('NYSE')  # Get the NYSE trading calendar
    
        # Determine the start (Monday) and end (Friday) of the current week
        week_start = today - pd.Timedelta(days=today.weekday())
        week_end = week_start + pd.Timedelta(days=4)
    
        # Get valid trading days for the week from the NYSE calendar
        trading_days = nyse.valid_days(start_date=week_start, end_date=week_end)
        trading_days = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in trading_days]
    
        # Return the first trading day of the week
        return trading_days[0]
    
    # Define the trade start and end dates
    trade_start_date = find_trading_day(today_date)  # Start on the first trading day of the week
    trade_end_date = today_date  # End on today's date
    
    # Function to fetch historical price data from Yahoo Finance
    def fetch_prices(tickers, start_date, end_date):
        # Download stock data for the specified date range
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        open_prices = data['Open']  # Extract the open prices
        close_prices = data['Close']  # Extract the close prices
        return open_prices, close_prices
    
    # Fetch open and close prices for the optimized tickers
    open_prices, close_prices = fetch_prices(optimized_tickers, trade_start_date, trade_end_date)
    
    # Dictionary to track portfolio positions and daily values
    portfolio_data = {
        "Date": [],  # List of dates
        "PortfolioValue": [],  # List of portfolio values
        "Positions": {ticker: [] for ticker in optimized_tickers},  # Positions for each ticker
        "ClosePrices": {ticker: [] for ticker in optimized_tickers},  # Close prices for each ticker
    }
    
    # Initialize positions and cash
    positions = {}
    initial_cash = INITIAL_CASH  # Define initial cash amount
    
    # Calculate the number of shares to purchase for each ticker
    start_open_prices = open_prices.loc[trade_start_date]  # Get open prices on the start date
    for ticker, weight in zip(portfolio['Ticker'], portfolio['Weight']):
        num_shares = (initial_cash * weight) / start_open_prices[ticker]  # Allocate cash based on weights
        positions[ticker] = num_shares  # Store the calculated number of shares
    
    # Update portfolio value for each trading day
    for date in close_prices.loc[trade_start_date:trade_end_date].index:
        daily_close_prices = close_prices.loc[date]  # Get close prices for the day
        portfolio_value = sum(positions[ticker] * daily_close_prices[ticker] for ticker in optimized_tickers)  # Calculate total portfolio value
    
        # Append daily data to the portfolio dictionary
        portfolio_data["Date"].append(date)
        portfolio_data["PortfolioValue"].append(portfolio_value)
        for ticker in optimized_tickers:
            portfolio_data["Positions"][ticker].append(positions[ticker])
            portfolio_data["ClosePrices"][ticker].append(daily_close_prices[ticker])
    
    # Save portfolio values to a CSV file for visualization
    portfolio_df = pd.DataFrame({
        "Date": portfolio_data["Date"],
        "PortfolioValue": portfolio_data["PortfolioValue"],
    })
    portfolio_df.to_csv(portfolio_values_path, index=False)  # Save to the specified path
    
    # Save positions and close prices to another CSV file
    positions_df = pd.DataFrame(portfolio_data["Positions"])  # Convert positions to DataFrame
    positions_df["Date"] = portfolio_data["Date"]  # Add the date column
    close_prices_df = pd.DataFrame(portfolio_data["ClosePrices"])  # Convert close prices to DataFrame
    close_prices_df["Date"] = portfolio_data["Date"]  # Add the date column
    
    # Merge positions and close prices into a single DataFrame
    positions_and_prices_df = positions_df.merge(close_prices_df, on="Date", suffixes=("_Positions", "_ClosePrices"))
    positions_and_prices_df.to_csv(portfolio_positions_path, index=False)  # Save merged data to the specified path

# Function to create HTML
def generate_portfolio_dashboard(**kwargs):
    # Define file paths for input CSV files and the output HTML file
    base_path = os.path.expanduser("~/airflow/dags")
    values_csv = os.path.join(base_path, "portfolio_values.csv")
    positions_csv = os.path.join(base_path, "portfolio_positions.csv")
    output_html = os.path.join(base_path, "portfolio_dashboard.html")
    
    # Check if the CSV files exist
    if not os.path.exists(values_csv) or not os.path.exists(positions_csv):
        print("Required files do not exist")
        html_content = "<h1>No portfolio data available yet.</h1>"
        with open(output_html, "w") as f:
            f.write(html_content)
        exit()
    
    # Read the portfolio values and positions
    try:
        pv_df = pd.read_csv(values_csv, parse_dates=["Date"])  # Portfolio values
        positions_and_prices_df = pd.read_csv(positions_csv)  # Positions and close prices
    except Exception as e:
        print(f"Error reading CSV: {e}")
        html_content = "<h1>Error reading portfolio data.</h1>"
        with open(output_html, "w") as f:
            f.write(html_content)
        exit()
    
    # Validate that the data is non-empty and contains necessary columns
    if pv_df.empty or positions_and_prices_df.empty or 'PortfolioValue' not in pv_df.columns:
        print("No valid data available.")
        html_content = "<h1>No portfolio data available yet.</h1>"
        with open(output_html, "w") as f:
            f.write(html_content)
        exit()
    
    # Sort by date
    pv_df.sort_values("Date", inplace=True)
    
    # Retrieve the initial portfolio value
    try:
        initial_val = pv_df['PortfolioValue'].iloc[0]
    except IndexError:
        html_content = "<h1>No portfolio data available yet.</h1>"
        with open(output_html, "w") as f:
            f.write(html_content)
        exit()
    
    # Ensure all dates in the DataFrame are set to 20:00 for consistency
    pv_df['Date'] = pv_df['Date'].apply(lambda d: d.replace(hour=20, minute=0, second=0, microsecond=0))
    
    # Calculate cumulative returns over time
    pv_df['CumulativeReturn'] = pv_df['PortfolioValue'] / initial_val - 1
    
    # Retrieve the latest and previous entries for calculating changes
    latest_entry = pv_df.iloc[-1]
    previous_entry = pv_df.iloc[-2] if len(pv_df) > 1 else latest_entry
    
    latest_value = latest_entry['PortfolioValue']  # Latest portfolio value
    previous_value = previous_entry['PortfolioValue']  # Previous portfolio value
    change = latest_value - previous_value  # Daily change in portfolio value
    change_percentage = (change / previous_value) * 100 if previous_value != 0 else 0  # Change in percentage
    
    # Determine the earliest date from the DataFrame (now all at 20:00)
    first_date = pv_df['Date'].min()
    
    # Format for ticks: Time on top, Date on bottom
    tick_format = "%H:%M\n%b %d, %Y"
    
    # Portfolio Value Figure
    fig_portfolio_value = go.Figure()
    fig_portfolio_value.add_trace(go.Scatter(
        x=pv_df['Date'], 
        y=pv_df['PortfolioValue'],
        mode='lines', 
        name='Portfolio Value',
        line=dict(color='green' if pv_df['PortfolioValue'].iloc[-1] > 0 else 'red')
    ))
    fig_portfolio_value.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value',
        hovermode='x unified',
        plot_bgcolor='white'
    )
    fig_portfolio_value.update_xaxes(
        tickformat=tick_format,
        dtick=24*60*60*1000,  # one-day interval in ms
        tick0=first_date
    )
    
    # Cumulative Return Figure
    fig_cumulative_return = go.Figure()
    fig_cumulative_return.add_trace(go.Scatter(
        x=pv_df['Date'], 
        y=pv_df['CumulativeReturn'],
        mode='lines', 
        name='Cumulative Return',
        line=dict(color='green' if pv_df['CumulativeReturn'].iloc[-1] > 0 else 'red')
    ))
    fig_cumulative_return.update_layout(
        title='Cumulative Return Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        hovermode='x unified',
        plot_bgcolor='white'
    )
    fig_cumulative_return.update_xaxes(
        tickformat=tick_format,
        dtick=24*60*60*1000,
        tick0=first_date
    )
    
    portfolio_value_graphJSON = json.dumps(fig_portfolio_value, cls=plotly.utils.PlotlyJSONEncoder)
    cumulative_return_graphJSON = json.dumps(fig_cumulative_return, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Prepare stock performance details
    stock_details = []
    # Also set positions_and_prices_df times to 20:00 for consistency if it has dates
    if 'Date' in positions_and_prices_df.columns and not positions_and_prices_df.empty:
        positions_and_prices_df['Date'] = pd.to_datetime(positions_and_prices_df['Date'])
        positions_and_prices_df['Date'] = positions_and_prices_df['Date'].apply(lambda d: d.replace(hour=20, minute=0, second=0, microsecond=0))
    
    # Loop through tickers to gather performance details and generate graphs
    for ticker in [col for col in positions_and_prices_df.columns if col.endswith("_ClosePrices")]:
        # Extract details for each stock
        ticker_name = ticker.replace("_ClosePrices", "")
        shares_column = ticker_name + "_Positions"
        shares = positions_and_prices_df[shares_column].iloc[-1]
        
        # Skip tickers with negligible shares
        if shares > 1e-6:
            today_close = positions_and_prices_df[ticker].iloc[-1]
            yesterday_close = positions_and_prices_df[ticker].iloc[-2] if len(positions_and_prices_df) > 1 else today_close
            
            # Calculate price change and percentage
            price_change = today_close - yesterday_close
            price_change_percentage = (price_change / yesterday_close) * 100 if yesterday_close != 0 else 0
            
            # Generate stock performance graph
            stock_fig = go.Figure()
            stock_fig.add_trace(go.Scatter(
                x=positions_and_prices_df['Date'],
                y=positions_and_prices_df[ticker],
                mode='lines',
                name=f'{ticker_name} Price',
                line=dict(color='green' if price_change > 0 else 'red')
            ))
            stock_fig.update_layout(
                title=f'{ticker_name} Performance',
                xaxis_title='Date',
                yaxis_title='Close Price',
                hovermode='x unified',
                plot_bgcolor='white'
            )
            stock_fig.update_xaxes(
                tickformat=tick_format,
                dtick=24*60*60*1000,
                tick0=first_date
            )
    
            stock_graphJSON = json.dumps(stock_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Append stock details
            stock_details.append({
                'ticker': ticker_name,
                'shares': shares,
                'current_price': today_close,
                'price_change': price_change,
                'price_change_percentage': price_change_percentage,
                'graphJSON': stock_graphJSON
            })
    
    # Generate the HTML content for the dashboard
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Portfolio Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .title {{
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 20px;
            }}
            .portfolio-value {{
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .change {{
                font-size: 1.5rem;
                margin-bottom: 30px;
            }}
            .change.positive {{
                color: green;
            }}
            .change.negative {{
                color: red;
            }}
            .stock-section {{
                margin-top: 30px;
            }}
            .divider {{
                height: 2px;
                background-color: #d3d3d3;
                margin: 20px 0;
                opacity: 0.5;
            }}
            .stock-value {{
                font-size: 1.25rem;
            }}
            .stock-value.positive {{
                color: green;
            }}
            .stock-value.negative {{
                color: red;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row">
                <div class="col text-center">
                    <div class="title">Portfolio Performance Tracking</div>
                </div>
            </div>
            <div class="row">
                <div class="col text-center">
                    <div class="portfolio-value">${latest_value:,.2f}</div>
                </div>
            </div>
            <div class="row">
                <div class="col text-center">
                    <div class="change {'positive' if change >= 0 else 'negative'}">
                        ${change:,.2f} ({change_percentage:.2f}%) Today
                    </div>
                </div>
            </div>
            <div class="divider"></div>
            <div class="row">
                <div class="col">
                    <div id="portfolio-value-chart"></div>
                </div>
            </div>
            <div class="divider"></div>
            <div class="row">
                <div class="col">
                    <div id="cumulative-return-chart"></div>
                </div>
            </div>
            <div class="divider"></div>
            <div class="row stock-section">
                <div class="col">
                    <h3>Stocks</h3>
    """
    
    for stock in stock_details:
        html += f"""
                    <div>
                        <h4>{stock['ticker']}</h4>
                        <p class="stock-value {'positive' if stock['price_change_percentage'] >= 0 else 'negative'}">
                            Current Price: ${stock['current_price']:,.2f}</p>
                        <p class="stock-value {'positive' if stock['price_change_percentage'] >= 0 else 'negative'}">
                            Change: ${stock['price_change']:,.2f} ({stock['price_change_percentage']:.2f}%)
                        </p>
                        <div id="{stock['ticker']}-chart"></div>
                    </div>
                    <div class="divider"></div>
                    <script>
                        var fig = {stock['graphJSON']};
                        Plotly.newPlot('{stock['ticker']}-chart', fig.data, fig.layout);
                    </script>
        """
    
    html += f"""
                </div>
            </div>
        </div>
        <script>
            var portfolioValueFig = {portfolio_value_graphJSON};
            Plotly.newPlot('portfolio-value-chart', portfolioValueFig.data, portfolioValueFig.layout);
    
            var cumulativeReturnFig = {cumulative_return_graphJSON};
            Plotly.newPlot('cumulative-return-chart', cumulativeReturnFig.data, cumulativeReturnFig.layout);
        </script>
    </body>
    </html>
    """
    
    # Write the generated HTML to the output file
    with open(output_html, "w") as f:
        f.write(html)
    
    print(f"HTML saved as {output_html}.")


# Default DAG arguments
default_args = {
    'owner': 'hhhao1004',
    'depends_on_past': False,
    'email_on_failure': False,  
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Default DAG arguments
default_args = {
    'owner': 'hhhao1004',  # The owner of the DAG
    'depends_on_past': False,  # Tasks do not depend on previous task runs
    'email_on_failure': False,  # Disable email notifications on failure
    'email_on_retry': False,  # Disable email notifications on retry
    'retries': 1,  # Number of retry attempts if a task fails
    'retry_delay': timedelta(minutes=5),  # Delay between retries
}

# Define the DAG
with DAG(
    'project_dag',  # Name of the DAG
    default_args=default_args,  # Apply default arguments
    description='Calculate portfolio weight based on last Friday data',  # Description of the DAG
    schedule_interval='0 20 * * *',  # Schedule to run daily at 8 PM
    start_date=datetime(2024, 12, 19, tzinfo=pendulum.timezone("America/New_York")),  # Start date of the DAG with timezone
    catchup=False,  # Disable backfilling for past dates
) as dag:
    
    # Task to check if today is a trading day
    def branch_check(**kwargs):
        # Check if today is a trading day and return the appropriate task
        if check_trading_day(**kwargs):  # Function to check trading day
            return 'calculate_weight'  # Continue to calculate weight if it is a trading day
        else:
            return 'end_task'  # Skip to end task if not a trading day

    check_trading_day_task = BranchPythonOperator(
        task_id='check_trading_day',  # Unique ID for the task
        python_callable=branch_check,  # Function to execute for branching logic
    )
    
    # Task to calculate portfolio weight
    calculate_weight_task = PythonOperator(
        task_id='calculate_weight',  # Unique ID for the task
        python_callable=calculate_weight,  # Function to calculate portfolio weight
    )
    
    # Task to generate an HTML portfolio dashboard
    generate_portfolio_dashboard_task = PythonOperator(
        task_id='generate_portfolio_dashboard',  # Unique ID for the task
        python_callable=generate_portfolio_dashboard,  # Function to generate the HTML dashboard
    )
    
    # Dummy task to mark the end of the workflow
    end_task = DummyOperator(
        task_id='end_task',  # Unique ID for the end task
    )

    # Define task dependencies
    # Branching task determines the next step: either calculate weight or end the workflow
    check_trading_day_task >> [calculate_weight_task, end_task]

    # If weights are calculated, generate the portfolio dashboard and then end the workflow
    calculate_weight_task >> generate_portfolio_dashboard_task >> end_task
