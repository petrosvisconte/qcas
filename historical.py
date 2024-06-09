#!/usr/bin/env python3

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytz
from dateutil.relativedelta import relativedelta
import yfinance as yf
from scipy.stats import norm


# import functions from monte.py
from monte import importData, calculateEV, calculateMinRR, IronCondor, calculatePricePaths


def analyzeProbabilityInterval(hist_start_date, hist_end_date, ticker, memory, interval):
    # Subtract years defined by memory from the historical start date
    date_obj = datetime.strptime(hist_start_date, '%Y-%m-%d')
    new_date_obj = date_obj - relativedelta(years=memory)
    start_date = new_date_obj.strftime('%Y-%m-%d')
    end_date = hist_start_date

    # Import historical data required
    all_hist_data = importData(ticker, start_date, hist_end_date)

    # Define data structures to store price information
    within_interval = pd.DataFrame(columns=['Start Date', 'End Date', 'Trading Days', 'Interval', 'Ending Price'])
    exceeded_interval = pd.DataFrame(columns=['Start Date', 'End Date', 'Trading Days', 'Interval', 'Ending Price'])

    price_in = 0
    price_out = 0
    while end_date != hist_end_date:
        print(start_date, end_date)
        # Find the expiry date for the monthly options contract
        expiry = find_monthly_expiry(int(end_date[:4]), int(end_date[5:7]))

        # Find the number of trading days between the start and expiry date
        days = trading_days(end_date, expiry)
        #print(expiry, days)
        
        # Retrieve the historical data for the specified interval
        historical_data = all_hist_data.loc[start_date:end_date]

        # Calculate the price paths for the specified interval
        price_paths = calculatePricePaths(historical_data, days, 10000)

        # TODO: Modify ironCondorModel to access historical options data
        # Generate the optimal Iron Condor model
        #IronCondor = ironCondorModel(price_paths, interval, ticker, expiry)

        # Calculate price interval
        offset = (100-interval*100)/2
        sp_s = round(np.percentile(price_paths[-1], offset))
        sc_s = round(np.percentile(price_paths[-1], 100-offset))
        #print(sp_s, sc_s)

        # Check if the price on the expiry date is within the interval
        end_price = all_hist_data.loc[end_date:expiry]
        # Get ending price on last day of interval
        end_price = end_price['Close'].iloc[-1]
        #print(end_price)
        if end_price >= sp_s and end_price <= sc_s:
            price_in += 1
            #print("Price is within the interval")
            within_interval = pd.concat([within_interval, pd.DataFrame([{'Start Date': end_date, 'End Date': expiry, 
                                'Trading Days': days, 'Interval': (sp_s, sc_s), 'Ending Price': end_price}])], ignore_index=True)
        else:
            price_out += 1
            #print("Price is outside the interval")
            exceeded_interval = pd.concat([exceeded_interval, pd.DataFrame([{'Start Date': end_date, 'End Date': 
                                    expiry, 'Trading Days': days, 'Interval': (sp_s, sc_s), 'Ending Price': end_price}])], ignore_index=True)

        
        # Add a month to the end date
        date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        new_date_obj = date_obj + relativedelta(months=1)
        end_date = new_date_obj.strftime('%Y-%m-%d')
        # Add a month to the start date
        date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        new_date_obj = date_obj + relativedelta(months=1)
        start_date = new_date_obj.strftime('%Y-%m-%d')        

    print("Price within the interval: ", price_in)
    print("Price outside the interval: ", price_out)
    print("Probability of price within the interval: ", price_in/(price_in+price_out))
    print(exceeded_interval)
    print(within_interval)
    print("Average trading days within interval: ", within_interval['Trading Days'].mean())
    print("Average trading days outside interval: ", exceeded_interval['Trading Days'].mean())
    
    return

# Find the optimal Iron Condor strikes
def ironCondorModel(price_paths, interval, ticker, expiry):
    # Define Iron Condor strikes based on percentiles and round to the nearest integer
    offset = (100-interval*100)/2
    sp_s = round(np.percentile(price_paths, offset))
    sc_s = round(np.percentile(price_paths, 100-offset))
    lp_s = round(np.percentile(price_paths, 2.5)) 
    lc_s = round(np.percentile(price_paths, 97.5))
    
    # Retrieve the option chain for SPY
    try:
        tickr = yf.Ticker(ticker)
        opt = tickr.option_chain(expiry)
    except:
        tickr = yf.Ticker(ticker)
        opt = tickr.option_chain(expiry)

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

    return iron_condor

def find_monthly_expiry(year, month):
    """
    Calculate the expiry date of monthly options contracts, accounting for NYSE holidays.

    Parameters:
    year (int): The year as a four-digit number.
    month (int): The month as a number (1-12).

    Returns:
    datetime.date: The adjusted expiry date of the options contract.
    """
    # Create a calendar for NYSE
    nyse_calendar = mcal.get_calendar('NYSE')

    # Fetch the trading days for the year
    start_date = datetime(year, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(year, 12, 31, tzinfo=pytz.UTC)
    nyse_trading_days = nyse_calendar.valid_days(start_date=start_date, end_date=end_date)

    # Start with the first day of the month, make it timezone-aware
    third_friday = datetime(year, month, 1, tzinfo=pytz.UTC)

    # Find the first Friday of the month
    while third_friday.weekday() != 4:
        third_friday += timedelta(days=1)

    # Add 14 days to get to the third Friday
    third_friday += timedelta(days=14)

    # If the third Friday is not a trading day, move to the previous trading day
    if third_friday not in nyse_trading_days:
        third_friday -= timedelta(days=1)
        while third_friday not in nyse_trading_days:
            third_friday -= timedelta(days=1)

    # Convert the timezone-aware datetime to naive datetime in UTC for the return value
    return third_friday.astimezone(pytz.UTC).replace(tzinfo=None).date()


def trading_days(start_date, end_date):
    """
    Calculate the number of trading days between two dates.
    """
    # Create a calendar for NYSE
    nyse = mcal.get_calendar('NYSE')
    # Get the schedule for the NYSE calendar
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    # Generate the trading days
    trading_days = mcal.date_range(schedule, frequency='1D')
    
    # Return the number of trading days
    return len(trading_days)



def main():
    print(trading_days('2024-06-09', '2024-06-28'))
    analyzeProbabilityInterval('2000-01-01', '2024-05-01', 'SPY', 10, 0.90)



if __name__ == "__main__":
    main()