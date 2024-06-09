#!/usr/bin/env python3

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as mpl
import seaborn as sb
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor



class IronCondor:
    def __init__(self, sp_s, sc_s, lp_s, lc_s, sp_p, sc_p, lp_p, lc_p):
        self.sp_s = sp_s
        self.sc_s = sc_s
        self.lp_s = lp_s
        self.lc_s = lc_s
        self.sp_p = sp_p
        self.sc_p = sc_p
        self.lp_p = lp_p
        self.lc_p = lc_p
        self.margin = max((sp_s - lp_s) * 100.0, (lc_s - sc_s) * 100.0)
        self.max_profit = (sp_p + sc_p - lp_p - lc_p) * 100.0
        self.max_loss_p = self.max_profit - (sp_s - lp_s) * 100.0
        self.max_loss_c = self.max_profit - (lc_s - sc_s) * 100.0
        self.break_even_p = sp_s - (sp_p + lp_p)
        self.break_even_c = sc_s + (sc_p + lc_p)
        self.rr = self.max_profit / min(self.max_loss_p, self.max_loss_c)

def importData(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Expected value
def calculateEV(rr, interval):
    return abs(rr)*interval - 1*(1-interval)

# Minimum RR required to break even
def calculateMinRR(interval):
    return (1-interval)/interval

def calculatePricePaths(data, days, trials):
    log_return = np.log(1 + data.iloc[:, 1].pct_change())

    u = log_return.mean()
    var = log_return.var()
    drift = u - (0.5*var)

    stdev = log_return.std()
    Z = norm.ppf(np.random.rand(days, trials))
    daily_returns = np.exp(drift + stdev * Z)

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data.iloc[-1, 3]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1]*daily_returns[t]

    return price_paths

# Find the optimal Iron Condor strikes
def ironCondorModel(price_paths, interval, ticker):
    # Define Iron Condor strikes based on percentiles and round to the nearest integer
    offset = (100-interval*100)/2
    sp_s = round(np.percentile(price_paths[-1], offset))
    sc_s = round(np.percentile(price_paths[-1], 100-offset))
    lp_s = round(np.percentile(price_paths[-1], 2.5)) 
    lc_s = round(np.percentile(price_paths[-1], 97.5))
    
    # Retrieve the option chain for SPY
    try:
        tickr = yf.Ticker(ticker)
        opt = tickr.option_chain("2024-06-28")  # replace with your desired expiration date
    except:
        tickr = yf.Ticker(ticker)
        opt = tickr.option_chain("2024-06-28")  # replace with your desired expiration date

    # Retrieve the premium for each option contract
    try: 
        sc_p = opt.calls.loc[opt.calls['strike'] == sc_s, 'lastPrice'].values[0]
        sp_p = opt.puts.loc[opt.puts['strike'] == sp_s, 'lastPrice'].values[0]
        #lp_p = opt.puts.loc[opt.puts['strike'] == lp_s, 'lastPrice'].values[0]
        #lc_p = opt.calls.loc[opt.calls['strike'] == lc_s, 'lastPrice'].values[0]
    except:
        sc_p = opt.calls.loc[opt.calls['strike'] == sc_s, 'lastPrice'].values[0]
        sp_p = opt.puts.loc[opt.puts['strike'] == sp_s, 'lastPrice'].values[0]
        #lp_p = opt.puts.loc[opt.puts['strike'] == lp_s, 'lastPrice'].values[0]
        #lc_p = opt.calls.loc[opt.calls['strike'] == lc_s, 'lastPrice'].values[0]

    # Retrieve all possible strike prices less than the short put strike
    lp_s_possible = opt.puts[opt.puts['strike'] < sp_s]['strike'].values
    # Retrieve all possible strike prices greater than the short call strike
    lc_s_possible = opt.calls[opt.calls['strike'] > sc_s]['strike'].values
    
    best_rr = float('inf')
    iron_condor = None
    # Find the optimal outer contracts for the Iron Condor
    for lp_s in lp_s_possible:
        # Retrieve the premium for each option contract
        try: 
            lp_p = opt.puts.loc[opt.puts['strike'] == lp_s, 'lastPrice'].values[0]
        except:
            lp_p = opt.puts.loc[opt.puts['strike'] == lp_s, 'lastPrice'].values[0]
        for lc_s in lc_s_possible:
            # Retrieve the premium for each option contract
            try: 
                lc_p = opt.calls.loc[opt.calls['strike'] == lc_s, 'lastPrice'].values[0]
            except:
                lc_p = opt.calls.loc[opt.calls['strike'] == lc_s, 'lastPrice'].values[0]
            # Calculate the statistics for the Iron Condor
            ic = IronCondor(sp_s, sc_s, lp_s, lc_s, sp_p, sc_p, lp_p, lc_p)
            if ic.rr < best_rr:
                best_rr = ic.rr
                iron_condor = ic
    
    # Print the statistics for the optimal Iron Condor
    print("Asset: ", ticker)
    print("Optimal Iron Condor: " + str(int(interval*100)) + "% interval")
    print("Strike prices: ", iron_condor.lp_s, iron_condor.sp_s, iron_condor.sc_s, iron_condor.lc_s)
    print("Premiums: ", iron_condor.lp_p, iron_condor.sp_p, iron_condor.sc_p, iron_condor.lc_p)
    print("Margin required:", iron_condor.margin)
    print("Max profit: ", iron_condor.max_profit)
    print("Max loss - put: ", iron_condor.max_loss_p)
    print("Max loss - call: ", iron_condor.max_loss_c)
    print("Break-even points: ", iron_condor.break_even_p, iron_condor.break_even_c)
    print("Risk-reward ratio: ", abs(iron_condor.rr))
    print("Minimum risk-reward ratio: ", calculateMinRR(interval))
    print("Expected value: ", calculateEV(iron_condor.rr, interval)*abs(min(iron_condor.max_loss_p, iron_condor.max_loss_c)))
    print("Expected value (as a ratio): ", calculateEV(iron_condor.rr, interval))
    print("\n")    

    return IronCondor

def main():
    INTERVAL = 0.8
    # ETFs
    #TICKER = 'SPY' # S&P 500
    #TICKER = 'QQQ' # Nasdaq
    #TICKER = 'IWM' # Russell 2000
    #TICKER = 'TLT' # 20+ Year Treasury Bond
    TICKER = 'SLV' # Silver
    #TICKER = 'GDX' # Gold Miners
    # Stocks

    data = importData(TICKER, '2000-01-01', '2024-06-06')
    #print(data.head(5))
    data.iloc[:, 3].plot(figsize=(10,5))
    #print(data.iloc[:, 3].head(5))
    mpl.xlabel("Time")
    mpl.ylabel("Price")
    mpl.title(TICKER + " Historical Price Data")
    mpl.show()

    log_return = np.log(1 + data.iloc[:, 1].pct_change())
    
    sb.displot(log_return.iloc[1:], bins=100, kde=True)
    mpl.xlabel("Daily Return")
    mpl.ylabel("Frequency")
    mpl.title("Distribution of Daily Log Returns for " + TICKER)
    mpl.show()

    # Calculate the price paths for the next 14 days
    price_paths = calculatePricePaths(data, 14, 10000)
    
    # Print current price:
    current_price = data.iloc[-1, 3]
    print("Current", TICKER, "price:", current_price)

    # Calculate the inner interval (dictates the range where the Iron Condor will be profitable)
    lower_bound = np.percentile(price_paths[-1], (100-INTERVAL*100)/2)
    upper_bound = np.percentile(price_paths[-1], 100-(100-INTERVAL*100)/2)
    # Calculate the 95% confidence interval
    lower_bound_1 = np.percentile(price_paths[-1], 2.5)
    upper_bound_1 = np.percentile(price_paths[-1], 97.5)
    # Calculate the 99% confidence interval
    lower_bound_2 = np.percentile(price_paths[-1], 0.5)
    upper_bound_2 = np.percentile(price_paths[-1], 99.5)
    print(str(int(INTERVAL*100)) + "% interval: ", lower_bound, upper_bound)
    print("95% interval: ", lower_bound_1, upper_bound_1)
    print("99% interval: ", lower_bound_2, upper_bound_2)
    print("\n")

    # # Plotting the predicted price as a probability distribution
    # sb.displot(price_paths[-1], bins=50, color='blue', legend=False, kde=True)
    # mpl.axvline(x=lower_bound, color='g', linestyle='--')  # Add a vertical line at the lower bound
    # mpl.axvline(x=upper_bound, color='g', linestyle='--')  # Add a vertical line at the upper bound
    # mpl.axvline(x=lower_bound_1, color='r', linestyle='--')  # Add a vertical line at the lower bound
    # mpl.axvline(x=upper_bound_1, color='r', linestyle='--')  # Add a vertical line at the upper bound
    # mpl.xlabel('Price')
    # mpl.ylabel('Frequency')
    # mpl.title('Prediction Price Distribution - After 100 Days', pad=0)
    # mpl.show()

    # # Download the actual future data
    # data_future = yf.download(TICKER, '2024-05-10', '2024-05-31')
    # data_future = data_future.iloc[:, 3]

    # # Plot the simulated price paths and the actual future prices on the same plot
    # mpl.figure(figsize=(10,5))
    # mpl.plot(price_paths)
    # mpl.plot(data_future.values, 'r', label='Actual', linewidth=4)
    # mpl.xlabel('Time (Days)')
    # mpl.ylabel('Price')
    # mpl.title('Simulated Price Paths')
    # mpl.legend()
    # mpl.show()

    #print("Minimum risk-reward ratio: ", calculateMinRR(INTERVAL), "\n")

    # Run the Iron Condor model
    #ironCondorModel(price_paths, INTERVAL, TICKER)
    # Run the Iron Condor model with a 95% interval
    #ironCondorModel(price_paths, 0.95, TICKER)
    # Run the Iron Condor model with a 90% interval
    ironCondorModel(price_paths, 0.9, TICKER)
    # Run the Iron Condor model with a 85% interval
    ironCondorModel(price_paths, 0.85, TICKER)
    # Run the Iron Condor model with a 80% interval
    ironCondorModel(price_paths, 0.8, TICKER)
    # Run the Iron Condor model with a 75% interval
    ironCondorModel(price_paths, 0.75, TICKER)
    # Run the Iron Condor model with a 70% interval
    ironCondorModel(price_paths, 0.7, TICKER)
    # Run the Iron Condor model with a 65% interval
    ironCondorModel(price_paths, 0.65, TICKER)
    # Run the Iron Condor model with a 60% interval
    ironCondorModel(price_paths, 0.6, TICKER)

    # # parallelize the Iron Condor model
    # intervals = [INTERVAL, 0.95, 0.9, 0.85, 0.75, 0.7]
    # with ThreadPoolExecutor() as executor:
    #     for interval in intervals:
    #         executor.submit(ironCondorModel, price_paths, interval)

if __name__ == '__main__':
    main()