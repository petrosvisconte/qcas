#!/usr/bin/env python3

import yfinance as yf
from dateutil.relativedelta import relativedelta


# import functions from monte.py and historical.py
from monte import IronCondor, ironCondorModel
from historical import analyzeProbabilityInterval


def main():
    print("=============================================")
    print("QCAS - Quantitative Condor Analytical System")
    print("Author: Pierre Visconti")
    print("Github: PetrosVisconte")
    print("https://github.com/petrosvisconte/qcas")
    print("=============================================")

    memory = 10
    assets = {'SPY': None, 'QQQ': None, 'IWM': None, 'TLT': None, 'SLV': None, 'GDX': None}
    
    # Determine the oldest possible date for historical data
    for asset in assets:
        data = yf.Ticker(asset)
        hist_data = data.history(period="max", progress=False)
        oldest_date = hist_data.index.min()
        oldest_date = oldest_date.replace(month=1, day=1) + relativedelta(years=memory+1)
        assets[asset] = oldest_date.strftime('%Y-%m-%d')

    for asset in assets:
        print("Analyzing asset: ", asset, assets[asset])
        analyzeProbabilityInterval(assets[asset], '2024-05-01', asset, memory, 0.9)



if __name__ == "__main__":
    main()


### Ideas:
# 1. Create a backtest analysis for longest possible historical data once a month
# 2. Create a backtest analysis for last 10 years with constant interval testing, for interval of length 14 and 13
# 3. Create a backtest analysis for last 5 years with constant interval testing, for interval of length 14 and 13
# 4. Create a backtest analysis for last 3 years with constant interval testing, for interval of length 14 and 13
# 5. Create a backtest analysis for last year with constant interval testing, for interval of length 14 and 13
# 6. Create a backtest analysis for last 6 months with constant interval testing, for interval of length 14 and 13
# 7. Create a backtest analysis for last 3 months with constant interval testing, for interval of length 14 and 13
# 8. Create a backtest analysis for last month with constant interval testing, for interval of length 14 and 13