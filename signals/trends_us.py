from datetime import datetime, timedelta
import json
import pandas as pd
import trends
import time

# print time
t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'{t}')

# get English stock ticker names (us, gb, au...)
names = trends.names

# restrict to US stock tickers
names = names[names['market_country'] == 'us']
print(f'{len(names)} US tickers to download trends for.')

# import existing trends csv
print(f'Importing trends_us.csv...', end='', flush=True)
df = pd.read_csv('trends_us.csv', parse_dates = ['date'], index_col='date')
print(f'{len(df)} rows already obtained across {len(df["ticker"].unique())} tickers. {len(names)-len(df["ticker"].unique())} remaining.')

i = 1
to_concat = []
tickers = [ticker for ticker in names.index if ticker not in df['ticker'].unique()]
tickers_len = len(tickers)

for ticker in tickers:
    print(f'{i}/{tickers_len}')
    keyword_list, column_names = trends.keyword_generator_3(ticker)
    trend_history = trends.ticker_trend(ticker, keyword_list, column_names)
    to_concat.append(trend_history)

    # export CSV every 5
    if i % 5 == 0:
        print('Exporting...')
        pd.concat([df] + to_concat).to_csv('trends_us.csv')
        print('Done.')

    # sleep every 5
    if i % 5 == 0:
        print('Sleeping for 60 seconds...', end='', flush=True)
        time.sleep(60)
        print('Woken!')

    i += 1
